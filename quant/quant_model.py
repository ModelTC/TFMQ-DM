from typing import List
import torch.nn as nn
import torch
from ldm.modules.attention import BasicTransformerBlock
from quant.quant_block import QuantAttentionBlock, QuantAttnBlock, QuantQKMatMul, QuantResnetBlock, QuantSMVMatMul, QuantTemporalInformationBlock, QuantTemporalInformationBlockDDIM, b2qb, BaseQuantBlock
from quant.quant_block import QuantBasicTransformerBlock, QuantResBlock
from quant.quant_layer import QMODE, QuantLayer, StraightThrough


class QuantModel(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 wq_params: dict = {},
                 aq_params: dict = {},
                 cali: bool = True,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.model = model
        self.softmax_a_bit = kwargs.get("softmax_a_bit", 8)
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        self.B = b2qb(aq_params['leaf_param'])
        self.quant_module(self.model, wq_params, aq_params, aq_mode=kwargs.get("aq_mode", [QMODE.NORMAL.value]), prev_name=None)
        self.quant_block(self.model, wq_params, aq_params)
        if cali:
            self.get_tib(self.model, wq_params, aq_params)

    def get_tib(self,
                    module: nn.Module,
                    wq_params: dict = {},
                    aq_params: dict = {},
                    ) -> QuantTemporalInformationBlock:
        for name, child in module.named_children():
            if name == 'temb':
                self.tib = QuantTemporalInformationBlockDDIM(child, aq_params, self.model.ch)
            elif name == 'time_embed':
                self.tib = QuantTemporalInformationBlock(child, aq_params, self.model.model_channels, None)
            elif isinstance(child, QuantResBlock):
                self.tib.add_emb_layer(child.emb_layers)
            elif isinstance(child, QuantResnetBlock):
                self.tib.add_temb_proj(child.temb_proj)
            else:
                self.get_tib(child, wq_params, aq_params)


    def quant_module(self,
                     module: nn.Module,
                     wq_params: dict = {},
                     aq_params: dict = {},
                     aq_mode: List[int] = [QMODE.NORMAL.value],
                     prev_name: str = None,
                     ) -> None:
        for name, child in module.named_children():
            if isinstance(child, tuple(QuantLayer.QMAP.keys())) and \
                'skip' not in name and 'op' not in name and not (prev_name == 'downsample' and name == 'conv') and 'shortcut' not in name: # refer to PTQD
                if prev_name is not None and 'emb_layers' in prev_name and '1' in name or 'temb_proj' in name:
                    setattr(module, name, QuantLayer(child, wq_params, aq_params, aq_mode=aq_mode, quant_emb=True))
                    continue
                setattr(module, name, QuantLayer(child, wq_params, aq_params, aq_mode=aq_mode))
            elif isinstance(child, StraightThrough):
                continue
            else:
                self.quant_module(child, wq_params, aq_params, aq_mode=aq_mode, prev_name=name)

    def quant_block(self,
                    module: nn.Module,
                    wq_params: dict = {},
                    aq_params: dict = {},
                    ) -> None:
        for name, child in module.named_children():
            if child.__class__.__name__ in self.B:
                if self.B[child.__class__.__name__] in [QuantBasicTransformerBlock, QuantAttnBlock]:
                    setattr(module, name, self.B[child.__class__.__name__](child, aq_params, softmax_a_bit = self.softmax_a_bit))
                elif self.B[child.__class__.__name__] in [QuantResnetBlock, QuantAttentionBlock, QuantResBlock]:
                    setattr(module, name, self.B[child.__class__.__name__](child, aq_params))
                elif self.B[child.__class__.__name__] in [QuantSMVMatMul]:
                    setattr(module, name, self.B[child.__class__.__name__](aq_params, softmax_a_bit = self.softmax_a_bit))
                elif self.B[child.__class__.__name__] in [QuantQKMatMul]:
                    setattr(module, name, self.B[child.__class__.__name__](aq_params))
            else:
                self.quant_block(child, wq_params, aq_params)

    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        for m in self.model.modules():
            if isinstance(m, (BaseQuantBlock, QuantLayer)):
                m.set_quant_state(use_wq=use_wq, use_aq=use_aq)

    def forward(self,
                x: torch.Tensor,
                timestep: int = None,
                context: torch.Tensor = None,
                ) -> torch.Tensor:
        if context is None:
            return self.model(x, timestep)
        return self.model(x, timestep, context)

    def disable_out_quantization(self) -> None:
        modules = []
        for m in self.model.modules():
            if isinstance(m, QuantLayer):
                modules.append(m)
        modules: List[QuantLayer]
        # disable the last layer and the first layer
        modules[-1].use_wq = False
        modules[-1].disable_aq = True
        modules[0].disable_aq = True
        modules[0].use_wq = False
        modules[1].disable_aq = True
        modules[2].disable_aq = True
        modules[2].use_wq = False
        modules[3].disable_aq = True
        modules[0].ignore_recon = True
        modules[2].ignore_recon = True
        modules[-1].ignore_recon = True

    def set_grad_ckpt(self, grad_ckpt: bool) -> None:
        for _, module in self.model.named_modules():
            if isinstance(module, (QuantBasicTransformerBlock, BasicTransformerBlock)):
                module.checkpoint = grad_ckpt

    def synchorize_activation_statistics(self):
        import linklink.dist_helper as dist
        for module in self.modules():
            if isinstance(module, QuantLayer):
                if module.aqtizer.delta is not None:
                    dist.allaverage(module.aqtizer.delta)


    def set_running_stat(self,
                         running_stat: bool = False
                         ) -> None:
        for m in self.model.modules():
            if isinstance(m, QuantBasicTransformerBlock):
                m.attn1.aqtizer_q.running_stat = running_stat
                m.attn1.aqtizer_k.running_stat = running_stat
                m.attn1.aqtizer_v.running_stat = running_stat
                m.attn1.aqtizer_w.running_stat = running_stat
                m.attn2.aqtizer_q.running_stat = running_stat
                m.attn2.aqtizer_k.running_stat = running_stat
                m.attn2.aqtizer_v.running_stat = running_stat
                m.attn2.aqtizer_w.running_stat = running_stat
            elif isinstance(m, QuantQKMatMul):
                m.aqtizer_q.running_stat = running_stat
                m.aqtizer_k.running_stat = running_stat
            elif isinstance(m, QuantSMVMatMul):
                m.aqtizer_v.running_stat = running_stat
                m.aqtizer_w.running_stat = running_stat
            elif isinstance(m, QuantAttnBlock):
                m.aqtizer_q.running_stat = running_stat
                m.aqtizer_k.running_stat = running_stat
                m.aqtizer_v.running_stat = running_stat
                m.aqtizer_w.running_stat = running_stat
            elif isinstance(m, QuantLayer):
                m.set_running_stat(running_stat)

