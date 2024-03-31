from typing import Dict, Tuple

from ldm.modules.diffusionmodules.util import timestep_embedding
from ddim.models.diffusion import AttnBlock, ResnetBlock, get_timestep_embedding, nonlinearity
from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, QKMatMul, ResBlock, SMVMatMul, TimestepBlock, checkpoint
from ldm.modules.attention import BasicTransformerBlock
import torch as th
import torch.nn as nn
from torch import einsum
from ldm.modules.attention import exists, default, CrossAttention
from einops import rearrange, repeat
from types import MethodType
from quant.quant_layer import QuantLayer, UniformAffineQuantizer, StraightThrough


class BaseQuantBlock(nn.Module):

    def __init__(self,
                 aq_params: dict = {}
                 ) -> None:
        super().__init__()
        self.use_wq = False
        self.use_aq = False
        self.act_func = StraightThrough()
        self.ignore_recon = False

    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(use_wq=use_wq, use_aq=use_aq)


class QuantTemporalInformationBlockDDIM(BaseQuantBlock):

    def __init__(self,
                 temb: nn.Module,
                 aq_params: dict = {},
                 ch: int = None
                 ) -> None:
        super().__init__(aq_params)
        self.temb = temb
        self.temb_projs = []
        self.ch = ch

    def add_temb_proj(self,
                      temb_proj: nn.Linear) -> None:
        self.temb_projs.append(temb_proj)

    def forward(self,
                x: th.Tensor,
                t: th.Tensor,
                ) -> Tuple[th.Tensor]:
        assert t is not None
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        opts = []
        for temb_proj in self.temb_projs:
            opts.append(temb_proj(nonlinearity(temb)))
        return tuple(opts)

    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(use_wq=use_wq, use_aq=use_aq)
        for temb_proj in self.temb_projs:
            assert isinstance(temb_proj, QuantLayer)
            temb_proj.set_quant_state(use_wq=use_wq, use_aq=use_aq)


class QuantTemporalInformationBlock(BaseQuantBlock):

    def __init__(self,
                 t_emb: nn.Sequential,
                 aq_params: dict = {},
                 model_channels: int = None,
                 num_classes: int = None
                 ) -> None:
        super().__init__(aq_params)
        self.t_emb = t_emb
        self.emb_layers = []
        self.label_emb_layer = None
        self.model_channels = model_channels
        self.num_classes = num_classes

    def add_emb_layer(self,
                      layer: nn.Sequential) -> None:
        self.emb_layers.append(layer)

    def add_label_emb_layer(self,
                            layer: nn.Sequential) -> None:
        self.label_emb = layer

    def forward(self,
                x: th.Tensor,
                t: th.Tensor,
                y: th.Tensor = None
                ) -> Tuple[th.Tensor]:
        assert t is not None
        t_emb = timestep_embedding(t, self.model_channels, repeat_only=False)
        emb = self.t_emb(t_emb)
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        opts = []
        for layer in self.emb_layers:
            opts.append(layer(emb))
        return tuple(opts)

    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        for m in self.modules():
            if isinstance(m, QuantLayer):
                m.set_quant_state(use_wq=use_wq, use_aq=use_aq)
        for emb_layer in self.emb_layers:
            for m in emb_layer.modules():
                if isinstance(m, QuantLayer):
                    m.set_quant_state(use_wq=use_wq, use_aq=use_aq)


# --------- Stable Diffusion Model -------- #
class QuantResBlock(BaseQuantBlock, TimestepBlock):
    def __init__(
        self,
        res: ResBlock,
        aq_params: dict = {}
        ) -> None:
        super().__init__(aq_params)
        self.channels = res.channels
        self.emb_channels = res.emb_channels
        self.dropout = res.dropout
        self.out_channels = res.out_channels
        self.use_conv = res.use_conv
        self.use_checkpoint = res.use_checkpoint
        self.use_scale_shift_norm = res.use_scale_shift_norm

        self.in_layers = res.in_layers

        self.updown = res.updown

        self.h_upd = res.h_upd
        self.x_upd = res.x_upd

        self.emb_layers = res.emb_layers
        self.out_layers = res.out_layers

        self.skip_connection = res.skip_connection

    def forward(self,
                x: th.Tensor,
                emb: th.Tensor = None,
                split: int = 0
                ) -> th.Tensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if split != 0 and hasattr(self.skip_connection, 'split') and self.skip_connection.split == 0:
            # resblock_updown use Identity() as skip_connection
            return checkpoint(
                self._forward, (x, emb, split), self.parameters(), self.use_checkpoint
            )
        return checkpoint(
                self._forward, (x, emb), self.parameters(), self.use_checkpoint
            )

    def _forward(self,
                 x: th.Tensor,
                 emb: th.Tensor,
                 split: int = 0
                 ) -> th.Tensor:
        if emb is None:
            assert(len(x) == 2)
            x, emb = x
        assert x.shape[2] == x.shape[3]

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        if split != 0:
            return self.skip_connection(x, split=split) + h
        return self.skip_connection(x) + h


def cross_attn_forward(self: CrossAttention,
                        x: th.Tensor,
                        context: th.Tensor = None,
                        mask: th.Tensor = None
                        ) -> th.Tensor:
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    if self.use_aq:
        quant_q = self.aqtizer_q(q)
        quant_k = self.aqtizer_k(k)
        sim = einsum('b i d, b j d -> b i j', quant_q, quant_k) * self.scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -th.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)
    attn = sim.softmax(dim=-1)

    if self.use_aq:
        out = einsum('b i j, b j d -> b i d', self.aqtizer_w(attn), self.aqtizer_v(v))
    else:
        out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


class QuantBasicTransformerBlock(BaseQuantBlock):
    def __init__(
        self,
        tran: BasicTransformerBlock,
        aq_params: dict = {},
        softmax_a_bit: int = 8
        ) -> None:
        super().__init__(aq_params)
        self.attn1 = tran.attn1
        self.ff = tran.ff
        self.attn2 = tran.attn2

        self.norm1 = tran.norm1
        self.norm2 = tran.norm2
        self.norm3 = tran.norm3
        self.checkpoint =  False

        self.attn1.aqtizer_q = UniformAffineQuantizer(**aq_params)
        self.attn1.aqtizer_k = UniformAffineQuantizer(**aq_params)
        self.attn1.aqtizer_v = UniformAffineQuantizer(**aq_params)

        self.attn2.aqtizer_q = UniformAffineQuantizer(**aq_params)
        self.attn2.aqtizer_k = UniformAffineQuantizer(**aq_params)
        self.attn2.aqtizer_v = UniformAffineQuantizer(**aq_params)

        aq_params_w = aq_params.copy()
        aq_params_w['bits'] = softmax_a_bit
        aq_params_w['symmetric'] = False
        aq_params_w['always_zero'] = True
        self.attn1.aqtizer_w = UniformAffineQuantizer(**aq_params_w)
        self.attn2.aqtizer_w = UniformAffineQuantizer(**aq_params_w)
        self.attn1.forward = MethodType(cross_attn_forward, self.attn1)
        self.attn2.forward = MethodType(cross_attn_forward, self.attn2)
        self.attn1.use_aq = False
        self.attn2.use_aq = False

    def forward(self,
                x: th.Tensor,
                context: th.Tensor = None
                ) -> th.Tensor:
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self,
                 x: th.Tensor,
                 context: th.Tensor = None
                 ) -> th.Tensor:
        assert context is not None

        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context) + x
        x = self.ff(self.norm3(x)) + x
        return x


# --------- Latent Diffusion Model -------- #
class QuantQKMatMul(BaseQuantBlock):
    def __init__(
        self,
        aq_params: dict = {}
        ) -> None:
        super().__init__(aq_params)
        self.scale = None
        self.use_aq = False
        self.aqtizer_q = UniformAffineQuantizer(**aq_params)
        self.aqtizer_k = UniformAffineQuantizer(**aq_params)

    def forward(self,
                q: th.Tensor,
                k: th.Tensor
                ) -> th.Tensor:
        if self.use_aq:
            quant_q = self.aqtizer_q(q * self.scale)
            quant_k = self.aqtizer_k(k * self.scale)
            weight = th.einsum(
                "bct,bcs->bts", quant_q, quant_k
            )
        else:
            weight = th.einsum(
                "bct,bcs->bts", q * self.scale, k * self.scale
            )
        return weight


class QuantSMVMatMul(BaseQuantBlock):
    def __init__(
        self,
        aq_params: dict = {},
        softmax_a_bit: int = 8
        ) -> None:
        super().__init__(aq_params)
        self.use_aq = False
        self.aqtizer_v = UniformAffineQuantizer(**aq_params)
        aq_params_w = aq_params.copy()
        aq_params_w['bits'] = softmax_a_bit
        aq_params_w['symmetric'] = False
        aq_params_w['always_zero'] = True
        self.aqtizer_w = UniformAffineQuantizer(**aq_params_w)

    def forward(self,
                weight: th.Tensor,
                v: th.Tensor
                ) -> th.Tensor:
        if self.use_aq:
            a = th.einsum("bts,bcs->bct", self.aqtizer_w(weight), self.aqtizer_v(v))
        else:
            a = th.einsum("bts,bcs->bct", weight, v)
        return a


class QuantAttentionBlock(BaseQuantBlock):
    def __init__(
        self,
        attn: AttentionBlock,
        aq_params: dict = {}
        ) -> None:
        super().__init__(aq_params)
        self.channels = attn.channels
        self.num_heads = attn.num_heads
        self.use_checkpoint = attn.use_checkpoint
        self.norm = attn.norm
        self.qkv = attn.qkv

        self.attention = attn.attention

        self.proj_out = attn.proj_out

    def forward(self,
                x: th.Tensor
                ) -> th.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self,
                 x: th.Tensor
                 ) -> th.Tensor:
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


# --------- DDIM Model -------- #
class QuantResnetBlock(BaseQuantBlock):
    def __init__(
        self,
        res: ResnetBlock,
        aq_params: dict = {}
        ) -> None:
        super().__init__(aq_params)
        self.in_channels = res.in_channels
        self.out_channels = res.out_channels
        self.use_conv_shortcut = res.use_conv_shortcut

        self.norm1 = res.norm1
        self.conv1 = res.conv1
        self.temb_proj = res.temb_proj
        self.norm2 = res.norm2
        self.dropout = res.dropout
        self.conv2 = res.conv2
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = res.conv_shortcut
            else:
                self.nin_shortcut = res.nin_shortcut


    def forward(self,
                x: th.Tensor,
                temb: th.Tensor = None,
                split: int = 0
                ) -> None:
        if temb is None:
            assert(len(x) == 2)
            x, temb = x

        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            elif hasattr(self.nin_shortcut, 'split'):
                x = self.nin_shortcut(x, split=split)
            else:
                x = self.nin_shortcut(x)
        out = x + h
        return out


class QuantAttnBlock(BaseQuantBlock):
    def __init__(
        self,
        attn: AttnBlock,
        aq_params: dict = {},
        softmax_a_bit: int = 8
        ) -> None:
        super().__init__(aq_params)
        self.in_channels = attn.in_channels

        self.norm = attn.norm
        self.q = attn.q
        self.k = attn.k
        self.v = attn.v
        self.proj_out = attn.proj_out

        self.aqtizer_q = UniformAffineQuantizer(**aq_params)
        self.aqtizer_k = UniformAffineQuantizer(**aq_params)
        self.aqtizer_v = UniformAffineQuantizer(**aq_params)

        aq_params_w = aq_params.copy()
        aq_params_w['bits'] = softmax_a_bit
        aq_params_w['symmetric'] = False
        aq_params_w['always_zero'] = True
        self.aqtizer_w = UniformAffineQuantizer(**aq_params_w)


    def forward(self,
                x: th.Tensor
                ) -> th.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        if self.use_aq:
            q = self.aqtizer_q(q)
            k = self.aqtizer_k(k)
        w_ = th.bmm(q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)
        if self.use_aq:
            v = self.aqtizer_v(v)
            w_ = self.aqtizer_w(w_)
        h_ = th.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        out = x + h_
        return out


def b2qb(use_aq: bool = False) -> Dict[nn.Module, BaseQuantBlock]:
    D = {
        ResBlock.__name__: QuantResBlock,
        BasicTransformerBlock.__name__: QuantBasicTransformerBlock,
        ResnetBlock.__name__: QuantResnetBlock,
        AttnBlock.__name__: QuantAttnBlock,
    }
    if use_aq:
        D[QKMatMul.__name__] = QuantQKMatMul
        D[SMVMatMul.__name__] = QuantSMVMatMul
    else:
        D[AttentionBlock.__name__] = QuantAttentionBlock
    return D
