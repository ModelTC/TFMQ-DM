from typing import Any, Dict, Tuple, Union
from ldm.models.diffusion.ddpm import LatentDiffusion
import torch.nn as nn
import torch
import numpy as np
from quant.quant_block import BaseQuantBlock
from quant.quant_model import QuantModel
from quant.quant_layer import QuantLayer
from quant.adaptive_rounding import AdaRoundQuantizer, RMODE
from quant.quant_layer import UniformAffineQuantizer
from tqdm import trange

from quant.reconstruction import block_reconstruction, layer_reconstruction, tib_reconstruction
import linklink as dist
import logging
logger = logging.getLogger(__name__)


def uaq2adar(model: nn.Module):
    for _, child in model.named_children():
        if isinstance(child, QuantLayer):
            if not child.ignore_recon:
                child.wqtizer = AdaRoundQuantizer(child.wqtizer,
                                                rmode = RMODE.LEARNED_HARD_SIGMOID,
                                                w = child.original_w.data)
        elif isinstance(child, BaseQuantBlock):
            if not child.ignore_recon:
                for _, sub_child in child.named_modules():
                    if isinstance(sub_child, QuantLayer):
                        if not hasattr(sub_child, 'wqtizer1'):
                            sub_child.wqtizer = AdaRoundQuantizer(sub_child.wqtizer,
                                                                rmode = RMODE.LEARNED_HARD_SIGMOID,
                                                                w = sub_child.original_w.data)
                        else:
                            sub_child.wqtizer = AdaRoundQuantizer(sub_child.wqtizer,
                                                                rmode = RMODE.LEARNED_HARD_SIGMOID,
                                                                w = sub_child.original_w.data[:, :sub_child.split, ...])
                            sub_child.wqtizer1 = AdaRoundQuantizer(sub_child.wqtizer1,
                                                                rmode = RMODE.LEARNED_HARD_SIGMOID,
                                                                w = sub_child.original_w.data[:, sub_child.split:, ...])
        else:
            uaq2adar(child)


def cali_model(qnn: QuantModel,
                      w_cali_data: Tuple[torch.Tensor],
                      a_cali_data: Tuple[torch.Tensor],
                      use_aq: bool = False,
                      path: str = None,
                      running_stat: bool = False,
                      interval: int = 128,
                      **kwargs
                      ) -> None:
    logger.info("Calibrating...")

    def recon_model(model: nn.Module, tag: bool = False) -> bool:
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            logger.info(f'block name: {name} quant: {isinstance(module, BaseQuantBlock)}')
            if name == '0' and cali_data[0].shape[-1] == 64 and len(cali_data) == 3:
                kwargs['keep_gpu'] = False
            if name == 'output_blocks':
                tag = True
            if tag and name.isdigit() and int(name) >= 8:
                kwargs['keep_gpu'] = False
            if name == 'tib':
                continue
            if name == 'time_embed' or name == 'temb':
                logger.info('Reconstruction for time embedding')
                tib_reconstruction(qnn.tib, cali_data=cali_data, **kwargs)
                continue
            if isinstance(module, QuantLayer): 
                if not module.ignore_recon:
                    logger.info('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(qnn, module, cali_data=cali_data, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if not module.ignore_recon:
                    logger.info('Reconstruction for block {}'.format(name))
                    block_reconstruction(qnn, module, cali_data=cali_data, **kwargs)
            else:
                tag = recon_model(module, tag=tag)
        return tag

    # --------- weight initialization -------- #
    cali_data = w_cali_data
    qnn.set_quant_state(use_wq = True, use_aq = False)
    batch_size = min(8, cali_data[0].shape[0])
    inputs = (x[: batch_size].cuda() for x in cali_data) 
    qnn(*inputs)
    qnn.disable_out_quantization()

    # --------- weight quantization -------- #
    recon_model(qnn)
    qnn.set_quant_state(use_wq = True, use_aq = False)
    delattr(qnn, 'tib')

    for name, module in qnn.model.named_modules():
        if 'wqtizer' in name:
            if not torch.is_tensor(module.zero_point):
                module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
            else:
                module.zero_point = nn.Parameter(module.zero_point)
            module.delta = nn.Parameter(module.delta)
    model_dict = {'weight': qnn.cpu().state_dict()}

    if use_aq:
        qnn.cuda()
        qnn.eval()
        cali_data = a_cali_data
        for time in range(cali_data[0].shape[0] // interval):
            t_cali_data = tuple([x[time * interval: (time + 1) * interval] for x in cali_data])
            qnn.set_quant_state(use_wq = True, use_aq = True)
            for name, module in qnn.model.named_modules():
                if 'aqtizer' in name:
                    del module.delta
                    del module.zero_point
                    module.delta = None
                    module.zero_point = None
                    module.init = False

            batch_size = min(16, t_cali_data[0].shape[0])
            with torch.no_grad():
                inds = np.random.choice(t_cali_data[0].shape[0], 16, replace=False)
                inputs = (x[inds].cuda() for x in t_cali_data)
                _ = qnn(*inputs)
                if running_stat:
                    logger.info('running stat for activation calibration...')
                    inds = np.arange(t_cali_data[0].shape[0])
                    np.random.shuffle(inds)
                    qnn.set_running_stat(True)
                    for i in trange(0, t_cali_data[0].shape[0], batch_size):
                        inputs = (x[inds[i: i + batch_size]].cuda() for x in t_cali_data)
                        _ = qnn(*inputs)
                    qnn.set_running_stat(False)
                    logger.info('running stat for activation calibration done.')
                torch.cuda.empty_cache()
            for name, module in qnn.model.named_modules():
                if 'aqtizer' in name:
                    if isinstance(module, UniformAffineQuantizer) and module.delta is not None:
                        if not torch.is_tensor(module.zero_point):
                            module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                        else:
                            module.zero_point = nn.Parameter(module.zero_point)

            temp = {}
            for name, module in qnn.model.named_modules():
                if 'aqtizer' in name and len(list(module.cpu().state_dict().keys())) == 2:
                    temp['model.' + name + '.delta'] = module.cpu().state_dict()['delta']
                    temp['model.' + name + '.zero_point'] = module.cpu().state_dict()['zero_point']
            model_dict['act_{}'.format(time)] = temp
        if path:
            torch.save(model_dict, path)
    logger.info("Calibration done.")
    
    
def load_cali_model(qnn: QuantModel,
                    init_data: Tuple[torch.Tensor],
                    use_aq: bool = False,
                    path: str = None,
                    ) -> None:
    
    logger.info("Loading calibration model...")
    
    ckpt = torch.load(path, map_location='cpu')['weight']
    qnn.set_quant_state(use_wq = True, use_aq = False)
    _ = qnn(*(_.cuda() for _ in init_data))
    qnn.disable_out_quantization()
    for key in ckpt.keys():
        if 'alpha' in key: # have used brecq
            uaq2adar(qnn)
            break
        
    for name, module in qnn.model.named_modules():
        if "wqtizer" in name:
            if isinstance(module, (UniformAffineQuantizer, AdaRoundQuantizer)):
                if not torch.is_tensor(module.zero_point):
                    module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                else:
                    module.zero_point = nn.Parameter(module.zero_point)
                module.delta = nn.Parameter(module.delta)
            
    keys = [key for key in ckpt.keys() if "aqtizer" in key]
    for key in keys:
        del ckpt[key]
    qnn.load_state_dict(ckpt, strict=False) # TODO: recon ?
    qnn.set_quant_state(use_wq=True, use_aq=False)
    
    for name, module in qnn.model.named_modules():
        if 'wqtizer' in name:
            if isinstance(module, AdaRoundQuantizer):
                z = module.zero_point.data
                delattr(module, 'zero_point')
                module.zero_point = z 
                d = module.delta.data
                delattr(module, 'delta')
                module.delta = d
            if isinstance(module, UniformAffineQuantizer):
                z = module.zero_point.data
                delattr(module, 'zero_point')
                module.zero_point = z
            
    if use_aq:
        qnn.set_quant_state(use_wq=True, use_aq=True)
        _ = qnn(*(_.cuda() for _ in init_data))
        
        for module in qnn.model.modules():
            if isinstance(module, (UniformAffineQuantizer, AdaRoundQuantizer)) and module.delta is not None:
                if not torch.is_tensor(module.zero_point):
                    module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                else:
                    module.zero_point = nn.Parameter(module.zero_point)
            if isinstance(module, AdaRoundQuantizer):
                module.delta = nn.Parameter(module.delta)
        for module in qnn.model.modules():
            if isinstance(module, AdaRoundQuantizer):
                z = module.zero_point.data
                delattr(module, 'zero_point')
                module.zero_point = z 
                d = module.delta.data
                delattr(module, 'delta')
                module.delta = d
    logger.info("Loading calibration model done.")


#  ------------- multi-gpu calibration -------------- #
def cali_model_multi(gpu: int,
                      dist_backend: str,
                      world_size: int,
                      dist_url: str,
                      rank: int,
                      ngpus_per_node: int,
                      model: LatentDiffusion,
                      use_aq: bool,
                      path: str,
                      w_cali_data: Tuple[torch.Tensor],
                      a_cali_data: Tuple[torch.Tensor],
                      interval: int, # samples per t
                      running_stat: bool,
                      kwargs: Dict[str, Any]
                      ) -> None:
    if gpu is not None:
        logger.info("Use GPU: {} for calibration.".format(gpu))
    rank = rank * ngpus_per_node + gpu
    dist.init_process_group(backend=dist_backend,
                            init_method=dist_url,
                            world_size=world_size,
                            rank=rank)
    torch.cuda.set_device(gpu)
    model.cuda()
    model.eval()
    qnn = QuantModel(model.diffusion_model,
                     wq_params=kwargs['wq_params'],
                     aq_params=kwargs['aq_params'],
                     softmax_a_bit=kwargs['softmax_a_bit'],
                     aq_mode=kwargs['aq_mode'])
    if 'no_grad_ckpt' in kwargs.keys() and kwargs['no_grad_ckpt']:
        qnn.set_grad_ckpt(False)
        del kwargs['no_grad_ckpt']
    del kwargs['wq_params']
    del kwargs['aq_params']
    del kwargs['softmax_a_bit']
    del kwargs['aq_mode']
    qnn.cuda()
    qnn.eval()
    torch.backends.cudnn.benchmark = False

    c = []
    for i in range(len(w_cali_data)):
        d = []
        for j in range(w_cali_data[i].shape[0] // interval):
            d.append(w_cali_data[i][j * interval + gpu * interval // world_size: j * interval + (gpu + 1) * interval // world_size])
        c.append(torch.cat(d, dim=0))
    w_cali_data = tuple(c)
    c = []
    for i in range(len(a_cali_data)):
        d = []
        for j in range(a_cali_data[i].shape[0] // interval):
            d.append(a_cali_data[i][j * interval + gpu * interval // world_size: j * interval + (gpu + 1) * interval // world_size])
        c.append(torch.cat(d, dim=0))
    a_cali_data = tuple(c)

    logger.info("Calibrating...")

    def recon_model(model: nn.Module) -> int:
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if rank == 0:
                logger.info(f'block name: {name} quant: {isinstance(module, BaseQuantBlock)}')
            if name == 'tib':
                continue
            if name == 'time_embed' or name == 'temb':
                if rank == 0:
                    logger.info('Reconstruction for time embedding')
                tib_reconstruction(qnn.tib, cali_data=cali_data, **kwargs)
                continue
            if isinstance(module, QuantLayer):
                if not module.ignore_recon:
                    if rank == 0:
                        logger.info('Reconstruction for layer {}'.format(name))
                    layer_reconstruction(qnn, module, cali_data=cali_data, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if not module.ignore_recon:
                    if rank == 0:
                        logger.info('Reconstruction for block {}'.format(name))
                    block_reconstruction(qnn, module, cali_data=cali_data, **kwargs)
            else:
                recon_model(module)

    # --------- weight initialization -------- #
    cali_data = w_cali_data
    qnn.set_quant_state(use_wq = True, use_aq = False)
    for name, module in qnn.model.named_modules():
        if 'wqtizer' in name:
            module: Union[UniformAffineQuantizer, AdaRoundQuantizer]
            module.init = False
    batch_size = min(64, cali_data[0].shape[0])
    inputs = (x[: batch_size].cuda() for x in cali_data)
    qnn(*inputs)
    qnn.disable_out_quantization()

    # --------- weight quantization -------- #
    recon_model(qnn)
    qnn.set_quant_state(use_wq = True, use_aq = False)
    delattr(qnn, 'tib')

    if rank == 0:
        for name, module in qnn.model.named_modules():
            if 'wqtizer' in name:
                if not torch.is_tensor(module.zero_point):
                    module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                else:
                    module.zero_point = nn.Parameter(module.zero_point)
                module.delta = nn.Parameter(module.delta)
        model_dict = {'weight': qnn.cpu().state_dict()}
    if use_aq:
        qnn.cuda()
        qnn.eval()
        cali_data = a_cali_data
        interval = interval // world_size
        for time in range(cali_data[0].shape[0] // interval):
            t_cali_data = tuple([x[time * interval: (time + 1) * interval] for x in cali_data])
            qnn.set_quant_state(use_wq = True, use_aq = True)
            for name, module in qnn.model.named_modules():
                if 'aqtizer' in name:
                    del module.delta
                    del module.zero_point
                    module.delta = None
                    module.zero_point = None
                    module.init = False

            batch_size = min(16, t_cali_data[0].shape[0])
            with torch.no_grad():
                inds = np.random.choice(t_cali_data[0].shape[0], 16, replace=False)
                inputs = (x[inds].cuda() for x in t_cali_data)
                _ = qnn(*inputs)
                if running_stat:
                    logger.info('running stat for activation calibration...')
                    inds = np.arange(t_cali_data[0].shape[0])
                    np.random.shuffle(inds)
                    qnn.set_running_stat(True)
                    for i in trange(0, t_cali_data[0].shape[0], batch_size):
                        inputs = (x[inds[i: i + batch_size]].cuda() for x in t_cali_data)
                        _ = qnn(*inputs)
                    qnn.set_running_stat(False)
                    logger.info('running stat for activation calibration done.')
                if ngpus_per_node > 1:
                    qnn.synchorize_activation_statistics()
                torch.cuda.empty_cache()
            for name, module in qnn.model.named_modules():
                if 'aqtizer' in name:
                    if isinstance(module, UniformAffineQuantizer) and module.delta is not None:
                        if not torch.is_tensor(module.zero_point):
                            module.zero_point = nn.Parameter(torch.tensor(float(module.zero_point)))
                        else:
                            module.zero_point = nn.Parameter(module.zero_point)
            if rank == 0:
                temp = {}
                for name, module in qnn.model.named_modules():
                    if 'aqtizer' in name and len(list(module.cpu().state_dict().keys())) == 2:
                        temp['model.' + name + '.delta'] = module.cpu().state_dict()['delta']
                        temp['model.' + name + '.zero_point'] = module.cpu().state_dict()['zero_point']
                model_dict['act_{}'.format(time)] = temp
        if path and rank == 0:
            torch.save(model_dict, path)
    logger.info("Calibration done.")