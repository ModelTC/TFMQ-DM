from quant.quant_block import BaseQuantBlock, QuantAttnBlock, QuantBasicTransformerBlock, QuantQKMatMul, QuantResBlock, QuantSMVMatMul, QuantTemporalInformationBlock
from quant.quant_layer import QMODE, QuantLayer, StraightThrough
from quant.quant_model import QuantModel
from quant.adaptive_rounding import AdaRoundQuantizer, RMODE
from quant.reconstruction_util import RLOSS, LossFuncTimeEmbedding
from quant.reconstruction_util import LossFunc
from typing import Tuple
from quant.data_utill import save_inout, save_grad
import torch
import linklink as link


def layer_reconstruction(model: QuantModel,
                         layer: QuantLayer,
                         cali_data: Tuple[torch.Tensor],
                         batch_size: int = 128,
                         iters: int = 20000,
                         w: float = 0.001,
                         opt_mode: RLOSS = RLOSS.MSE,
                         asym: bool = False,
                         include_act_func: bool = True,
                         b_range: tuple = (20, 2),
                         warmup: float = 0.0,
                         use_aq: bool = False,
                         lr: float = 4e-5,
                         p: float = 2.0,
                         multi_gpu: bool = False,
                         keep_gpu=True
                         ) -> None:
    model.set_quant_state(use_wq=False, use_aq=False)
    layer.set_quant_state(use_wq=True, use_aq=use_aq)
    if not include_act_func:
        org_act_func = layer.act_func
        layer.act_func = StraightThrough()

    if not use_aq:
        layer.wqtizer = AdaRoundQuantizer(uaqtizer=layer.wqtizer,
                                            rmode=RMODE.LEARNED_HARD_SIGMOID,
                                            w=layer.original_w.data)
        layer.wqtizer.soft_tgt = True
        opt_params = [layer.wqtizer.alpha]
        optimizer = torch.optim.Adam(opt_params)
        scheduler = None
    else:
        opt_params = [layer.aqtizer.delta]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0)
    loss_func = LossFunc(o=layer,
                         round_loss=RLOSS.NONE if use_aq else RLOSS.RELAXATION,
                         w=w,
                         max_count=iters,
                         rec_loss=opt_mode,
                         b_range=b_range,
                         decay_start=0.0,
                         warmup=warmup,
                         p=p)
    cached_inputs, cached_outputs = save_inout(model, layer, cali_data, asym, use_aq, batch_size, keep_gpu)
    if opt_mode != RLOSS.MSE:
        cached_grads = save_grad(model, layer, cali_data, asym, use_aq, batch_size, keep_gpu)
    else:
        cached_grads = None
    device = next(layer.parameters()).device
    for i in range(iters):
        idx = torch.randperm(cached_inputs[0].size(0))[: batch_size]
        cur_inputs = (x[idx].to(device=device) for x in cached_inputs) # ^x
        cur_outputs = cached_outputs[idx].to(device=device) # z
        cur_grads = cached_grads[idx].to(device=device) if opt_mode != RLOSS.MSE else None
        optimizer.zero_grad()
        out_quant = layer(*cur_inputs) # ^z
        err = loss_func(out_quant, cur_outputs, cur_grads)
        err.backward(retain_graph=True)
        if multi_gpu:
            for param in opt_params: # output layer does not use quantizer
                if param.grad is not None:
                    link.allreduce(param.grad)
        optimizer.step()
        if scheduler:
            scheduler.step()
    torch.cuda.empty_cache()
    layer.wqtizer.soft_tgt = False
    if not include_act_func:
        layer.act_func = org_act_func



def block_reconstruction(model: QuantModel,
                         block: BaseQuantBlock,
                         cali_data: torch.Tensor,
                         batch_size: int = 32,
                         iters: int = 20000,
                         w: float = 0.01,
                         opt_mode: RLOSS = RLOSS.MSE,
                         asym: bool = False,
                         include_act_func: bool = True,
                         b_range: tuple = (20, 2),
                         warmup: float = 0.0,
                         use_aq: bool = False,
                         lr: float = 4e-5,
                         p: float = 2.0,
                         multi_gpu: bool = True,
                         keep_gpu=True
                         ) -> None:
    model.set_quant_state(use_wq=False, use_aq=False)
    block.set_quant_state(use_wq=True, use_aq=use_aq)

    if not include_act_func:
        org_act_func = block.act_func
        block.act_func = StraightThrough()

    if not use_aq:
        opt_params = []
        for _, module in block.named_modules():
            if isinstance(module, QuantLayer) and module.quant_emb is False:
                # for shortcut in ResBlock or ResnetBlock, no single layer has shortcut
                if module.split != 0 and QMODE.QDIFF.value in module.aq_mode:
                    module.wqtizer = AdaRoundQuantizer(uaqtizer=module.wqtizer,
                                                        rmode=RMODE.LEARNED_HARD_SIGMOID,
                                                        w=module.original_w.data[:, :module.split, ...])
                    module.wqtizer.soft_tgt = True
                    module.wqtizer1 = AdaRoundQuantizer(uaqtizer=module.wqtizer1,
                                                        rmode=RMODE.LEARNED_HARD_SIGMOID,
                                                        w=module.original_w.data[:, module.split:, ...])
                    module.wqtizer1.soft_tgt = True
                    opt_params += [module.wqtizer.alpha, module.wqtizer1.alpha]
                else:
                    module.wqtizer = AdaRoundQuantizer(uaqtizer=module.wqtizer,
                                                        rmode=RMODE.LEARNED_HARD_SIGMOID,
                                                        w=module.original_w.data)
                    module.wqtizer.soft_tgt = True
                    opt_params.append(module.wqtizer.alpha)
        if len(opt_params) == 0: # for QuantSMVMatMul and QuantQKMatMul
            return
        optimizer = torch.optim.Adam(opt_params)
        scheduler = None
    else:
        opt_params = []
        for _, module in block.named_modules():
            if isinstance(module, QuantLayer) and module.quant_emb is False:
                if module.aqtizer.delta:
                    opt_params.append(module.aqtizer.delta)
                # for shortcut in ResBlock or ResnetBlock, no single layer has shortcut
                if hasattr(module, 'aqtizer1') and module.aqtizer1.delta is not None:
                    opt_params.append(module.aqtizer1.delta)
        A = []
        if isinstance(block, QuantBasicTransformerBlock):
            A = [block.attn1.aqtizer_q, block.attn1.aqtizer_k, block.attn1.aqtizer_v,\
                block.attn2.aqtizer_q, block.attn2.aqtizer_k, block.attn2.aqtizer_v]
            if block.attn1.aqtizer_w.level != (2 ** 16):
                A.append(block.attn1.aqtizer_w)
            if block.attn2.aqtizer_w.level != (2 ** 16):
                A.append(block.attn2.aqtizer_w)
        elif isinstance(block, QuantQKMatMul):
            A = [block.aqtizer_q, block.aqtizer_k]
        elif isinstance(block, QuantSMVMatMul):
            A = [block.aqtizer_v]
            if block.aqtizer_w.level != (2 ** 16):
                A.append(block.aqtizer_w)
        elif isinstance(block, QuantAttnBlock):
            A = [block.aqtizer_q, block.aqtizer_k, block.aqtizer_v]
            if block.aqtizer_w.level != (2 ** 16):
                A.append(block.aqtizer_w)
        for aqtizer in A:
            if aqtizer.delta:
                opt_params.append(aqtizer.delta)
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0)
    loss_func = LossFunc(o=block,
                         round_loss=RLOSS.NONE if use_aq else RLOSS.RELAXATION,
                         w=w,
                         max_count=iters,
                         rec_loss=opt_mode,
                         b_range=b_range,
                         decay_start=0.0,
                         warmup=warmup,
                         p=p)
    cached_inputs, cached_outputs = save_inout(model, block, cali_data, asym, use_aq, batch_size, keep_gpu)
    if opt_mode != RLOSS.MSE:
        cached_grads = save_grad(model, block, cali_data, asym, use_aq, batch_size, keep_gpu)
    else:
        cached_grads = None
    device = next(block.parameters()).device
    for i in range(iters):
        idx = torch.randperm(cached_inputs[0].size(0))[: batch_size]
        cur_inputs = (x[idx].to(device=device) for x in cached_inputs)
        cur_outputs = cached_outputs[idx].to(device=device)
        cur_grads = cached_grads[idx].to(device=device) if opt_mode != RLOSS.MSE else None
        optimizer.zero_grad()

        # ResBlock's split or ResnetBlock's split has been set in save_inout or even before, and cur_inputs does not contain split
        out_quant = block(*cur_inputs)
        err = loss_func(out_quant, cur_outputs, cur_grads)
        err.backward(retain_graph=True)
        if multi_gpu:
            for param in opt_params:
                link.allreduce(param.grad)
        optimizer.step()
        if scheduler:
            scheduler.step()
    torch.cuda.empty_cache()
    for _, module in block.named_modules():
        if isinstance(module, QuantLayer) and module.quant_emb is False:
            if module.split != 0 and QMODE.QDIFF.value in module.aq_mode:
                module.wqtizer.soft_tgt = False
                module.wqtizer1.soft_tgt = False
            else:
                module.wqtizer.soft_tgt = False

    if not include_act_func:
        block.act_func = org_act_func


def tib_reconstruction(block: BaseQuantBlock,
                                  cali_data: torch.Tensor,
                                  batch_size: int = 32,
                                  iters: int = 20000,
                                  w: float = 0.01,
                                  opt_mode: RLOSS = RLOSS.MSE,
                                  asym: bool = False,
                                  include_act_func: bool = True,
                                  b_range: tuple = (20, 2),
                                  warmup: float = 0.0,
                                  use_aq: bool = False,
                                  lr: float = 4e-5,
                                  p: float = 2.0,
                                  multi_gpu: bool = True,
                                  keep_gpu=True) -> None:
    block.set_quant_state(use_wq=True, use_aq=use_aq)

    if not include_act_func:
        org_act_func = block.act_func
        block.act_func = StraightThrough()

    if not use_aq:
        opt_params = []
        for _, module in block.named_modules():
            if isinstance(module, QuantLayer):
                module.wqtizer = AdaRoundQuantizer(uaqtizer=module.wqtizer,
                                                    rmode=RMODE.LEARNED_HARD_SIGMOID,
                                                    w=module.original_w.data)
                module.wqtizer.soft_tgt = True
                opt_params.append(module.wqtizer.alpha)
        if isinstance(block, QuantTemporalInformationBlock):
            for emb_layers in block.emb_layers:
                for _, module in emb_layers.named_modules():
                    if isinstance(module, QuantLayer):
                        module.wqtizer = AdaRoundQuantizer(uaqtizer=module.wqtizer,
                                                            rmode=RMODE.LEARNED_HARD_SIGMOID,
                                                            w=module.original_w.data)
                        module.wqtizer.soft_tgt = True
                        opt_params.append(module.wqtizer.alpha)
        else:
            for temb_proj in block.temb_projs:
                assert isinstance(temb_proj, QuantLayer)
                temb_proj.wqtizer = AdaRoundQuantizer(uaqtizer=temb_proj.wqtizer,
                                                      rmode=RMODE.LEARNED_HARD_SIGMOID,
                                                      w=temb_proj.original_w.data)
                temb_proj.wqtizer.soft_tgt = True
                opt_params.append(temb_proj.wqtizer.alpha)
        optimizer = torch.optim.Adam(opt_params)
        scheduler = None
    else:
        opt_params = []
        for _, module in block.named_modules():
            if isinstance(module, QuantLayer):
                if module.aqtizer.delta:
                    opt_params.append(module.aqtizer.delta)
        if isinstance(block, QuantTemporalInformationBlock):
            for emb_layers in block.emb_layers:
                for _, module in emb_layers.named_modules():
                    if isinstance(module, QuantLayer):
                        opt_params.append(module.aqtizer.delta)
        else:
            for temb_proj in block.temb_projs:
                assert isinstance(temb_proj, QuantLayer)
                opt_params.append(temb_proj.aqtizer.delta)
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0)
    loss_func = LossFuncTimeEmbedding(o=block,
                         round_loss=RLOSS.NONE if use_aq else RLOSS.RELAXATION,
                         w=w,
                         max_count=iters,
                         rec_loss=opt_mode,
                         b_range=b_range,
                         decay_start=0.0,
                         warmup=warmup,
                         p=p)
    cached_inputs, cached_outputs = save_inout(block, block, cali_data, asym, use_aq, batch_size, keep_gpu)
    assert opt_mode == RLOSS.MSE
    device = next(block.parameters()).device
    for i in range(iters):
        idx = torch.randperm(cached_inputs[0].size(0))[: batch_size]
        cur_inputs = (x[idx].to(device=device) for x in cached_inputs)
        cur_outputs = (x[idx].to(device=device) for x in cached_outputs)
        optimizer.zero_grad()
        out_quant = block(*cur_inputs)
        err = loss_func(out_quant, cur_outputs)
        err.backward(retain_graph=True)
        if multi_gpu:
            for param in opt_params:
                link.allreduce(param.grad)
        optimizer.step()
        if scheduler:
            scheduler.step()
    torch.cuda.empty_cache()
    for _, module in block.named_modules():
        if isinstance(module, QuantLayer):
            module.wqtizer.soft_tgt = False
    if isinstance(block, QuantTemporalInformationBlock):
        for emb_layers in block.emb_layers:
            for _, module in emb_layers.named_modules():
                if isinstance(module, QuantLayer):
                    module.wqtizer.soft_tgt = False
    else:
        for temb_proj in block.temb_projs:
            assert isinstance(temb_proj, QuantLayer)
            temb_proj.wqtizer.soft_tgt = False
    if not include_act_func:
        block.act_func = org_act_func






