import torch.nn as nn
import torch
from enum import Enum
from typing import List, Union
import torch.nn.functional as F
import logging
import numpy as np
logger = logging.getLogger(__name__)


# -------- quantization utils -------- #
class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def minmax(x: torch.Tensor,
            symmetric: bool = False,
            level: int = 256,
            always_zero: bool = False
            ) -> [torch.Tensor, torch.Tensor]:
    x_min, x_max = min(x.min().item(), 0), max(x.max().item(), 0)
    delta = torch.tensor(float(x_max - x_min) / (level - 1))
    if symmetric:
        x_min, x_max = -max(abs(x_min), x_max), max(abs(x_min), x_max)
        delta = torch.tensor(float(x_max - x_min) / (level - 2))
    if always_zero:
        delta = torch.tensor(float(x_max) / (level - 1))
    if delta < 1e-8:
        delta = 1e-8
    zero_point = torch.round(-x_min / delta) if not (symmetric or always_zero) else 0
    return torch.tensor(delta).type_as(x), zero_point


def mse(x: torch.Tensor,
        symmetric: bool = False,
        level: int = 256,
        always_zero: bool = False
        ) -> [torch.Tensor, torch.Tensor]:
    x_min, x_max = x.min().item(), x.max().item()
    delta, zero_point = None, None
    s = 1e+10
    for i in range(80):
        new_min = x_min * (1. - (i * 0.01))
        new_max = x_max * (1. - (i * 0.01))
        new_delta = torch.tensor(float(new_max - new_min) / (level - 1))
        if symmetric: 
            new_min, new_max = -max(abs(new_min), new_max), max(abs(new_min), new_max)
            new_delta = (new_max - new_min) / (level - 2)
        if always_zero:
            new_delta = torch.tensor(float(new_max) / (level - 1))
        new_zero_point = torch.round(-new_min / new_delta) if not (symmetric or always_zero) else 0
        NB, PB = -level // 2 if symmetric and not always_zero else 0,\
              level // 2 - 1 if symmetric and not always_zero else level - 1
        x_q = torch.clamp(torch.round(x / new_delta) + new_zero_point, NB, PB)
        x_dq = new_delta * (x_q - new_zero_point)
        new_s = lp_loss(x_dq, x, p=2.4, reduction=REDUCTION.ALL)
        if new_s < s:
            s = new_s
            delta, zero_point = new_delta, new_zero_point 
    return delta, zero_point


def kl(x: torch.Tensor,
       symmetric: bool = False,
       level: int = 256,
       always_zero: bool = False
       ) -> [torch.Tensor, torch.Tensor]:

    def to_hist_with_orig_bins(targ_hist, targ_bins, orig_hist, orig_bins):
        targ_v = 0.0
        targ_i = 0
        targ_bin = targ_bins[0]
        ret_hist = np.zeros_like(orig_hist)

        for i, orig_bin in enumerate(orig_bins[:-1]):
            if targ_bin <= orig_bin:
                if targ_i < len(targ_bins) - 1:
                    targ_v = targ_hist[targ_i]
                    targ_i += 1
                    targ_bin = targ_bins[targ_i]
                else:
                    targ_v = 0.0
                    targ_bin = orig_bin.max() + 1.0

            ret_hist[i] = targ_v
        return ret_hist

    min_kl = 1e5
    res_clip_ratio = 1.0
    np_x = x.clone().detach().cpu().numpy()
    ref_hist, ref_bins = np.histogram(np_x, bins=level, density=True)
    sumd = np.sum(np.diff(ref_bins))
    smooth_ref_hist = (ref_hist + 1e-5) / (1.0 + sumd * 1e-5)
    for clip_ratio in np.linspace(0.5, 1.0, 50):
        clip_range = [np.min(np_x) * clip_ratio, np.max(np_x) * clip_ratio]
        q_hist, q_bins = np.histogram(np.clip(np_x, clip_range[0], clip_range[1]), bins=level, density=True)
        c_q_hist = to_hist_with_orig_bins(q_hist, q_bins, ref_hist, ref_bins)
        c_q_hist = (c_q_hist + 1e-5) / (1.0 + sumd * 1e-5)
        kl_c_q = np.sum(smooth_ref_hist * np.log(smooth_ref_hist / c_q_hist))
        if kl_c_q < min_kl:
            min_kl = kl_c_q
            res_clip_ratio = clip_ratio
    x_min, x_max = np.min(np_x) * res_clip_ratio, np.max(np_x) * res_clip_ratio
    x_clone = torch.where(x < x_min, x_min, x.clone().detach())
    x_clone = torch.where(x > x_max, x_max, x_clone)
    return minmax(x_clone, symmetric, level, always_zero)


def hist(x: torch.Tensor,
        symmetric: bool = False,
        level: int = 256,
        always_zero: bool = False
        ) -> [torch.Tensor, torch.Tensor]:
    x_min, x_max = None, None
    np_x = x.clone().detach().cpu().numpy()
    data_max = max(-np.min(np_x), np.max(np_x))
    hist, _ = np.histogram(np_x, bins=level, range=(0, data_max), density=True)
    accum = 0
    threshold = 0.9996
    hist = hist.astype(np.float32) / hist.sum()
    for i in range(len(hist)):
        accum += hist[i]
        if accum >= threshold:
            clip_value = (i + 0.5) * (data_max / level)
            x_min, x_max = max(-clip_value, np.min(np_x)), min(clip_value, np.max(np_x))
            break
    x_clone = torch.where(x < x_min, x_min, x.clone().detach())
    x_clone = torch.where(x > x_max, x_max, x_clone)
    return minmax(x_clone, symmetric, level, always_zero)


class Scaler(Enum):
    MINMAX = minmax
    MSE = mse
    KL = kl
    HIST = hist


REDUCTION = Enum('REDUCTION', ('NONE', 'ALL'))


def lp_loss(pred: torch.Tensor, 
            tgt: torch.Tensor, 
            p: int = 2., 
            reduction: REDUCTION = REDUCTION.NONE
            ) -> torch.Tensor:
    if reduction == REDUCTION.NONE:
        return (pred - tgt).abs().pow(p).sum(1).mean()
    elif reduction == REDUCTION.ALL:
        return (pred - tgt).abs().pow(p).mean()
    else:
        raise NotImplementedError


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    
    def __init__(self, 
                 bits: int = 8,
                 symmetric: bool = False,
                 channel_wise: bool = False,
                 scaler: Scaler = Scaler.MINMAX,
                 leaf_param: bool = False,
                 always_zero: bool = False, # for softmax
                 quant_emb: bool = False
                 ) -> None:
        super().__init__()
        self.level = 2 ** bits
        self.symmetric = symmetric
        self.channel_wise = channel_wise
        self.scaler = scaler
        self.leaf_param = leaf_param
        if self.leaf_param:
            self.x_min, self.x_max = None, None
        self.running_stat = False
        self.always_zero = always_zero
        self.delta = None
        self.zero_point = None
        self.init = False
        self.quant_emb = quant_emb

    def _init_quantization_param(self, 
                                 x: torch.Tensor, 
                                 channel_wise: bool = False
                                 ) -> [torch.Tensor, torch.Tensor]:
        if channel_wise:
            N = x.shape[0]
            x_clone = x.clone().detach()
            x_max = x_clone.abs()
            for _ in range(len(x.shape) - 1):
                x_max = x_max.max(dim=-1)[0]
            delta, zero_point = x_max.clone(), x_max.clone()
            for c in range(N):
                delta[c], zero_point[c] = self._init_quantization_param(x_clone[c], channel_wise=False)
            D = {4: (-1, 1, 1, 1), 3: (-1, 1, 1), 2: (-1, 1)}
            delta = delta.view(D[len(x.shape)]) 
            zero_point = zero_point.view(D[len(x.shape)])
        else:
            if self.leaf_param:
                self.x_min, self.x_max = x.data.min(), x.data.max()
            delta, zero_point = self.scaler(x, self.symmetric, self.level, self.always_zero)
        return delta, zero_point
    
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        if not self.init:
            self.delta, self.zero_point = self._init_quantization_param(x, self.channel_wise)
            if self.leaf_param:
                self.delta = nn.Parameter(self.delta) 
            self.init = True

        if self.running_stat:
           self.act_momentum_update(x)

        NB, PB = -self.level // 2 if self.symmetric and not self.always_zero else 0, \
            self.level // 2 - 1 if self.symmetric and not self.always_zero else self.level - 1
        x_q = torch.clamp(ste_round(x / self.delta) + self.zero_point, NB, PB)
        x_dq = self.delta * (x_q - self.zero_point)
        return x_dq
    
    def act_momentum_update(self,
                            x: torch.Tensor,
                            act_range_momentum: float = 0.95
                            ) -> None:
        assert self.init
        assert self.leaf_param
        x_min = x.data.min()
        x_max = x.data.max()
        self.x_min = self.x_min * act_range_momentum + x_min * (1. - act_range_momentum)
        self.x_max = self.x_max * act_range_momentum + x_max * (1. - act_range_momentum)
        x_clone = torch.where(x < self.x_min, self.x_min, x.clone().detach())
        x_clone = torch.where(x > self.x_max, self.x_max, x_clone)
        x_clone[..., 0] = self.x_min
        x_clone[..., 1] = self.x_max
        delta, self.zero_point = Scaler.MINMAX(x_clone, self.symmetric, self.level, self.always_zero)
        self.delta = torch.nn.Parameter(delta)

    def bitwidth_refactor(self, 
                          bits: int = 8
                          ) -> None:
        self.level = 2 ** bits

    def extra_repr(self) -> str:
        s = 'level={level}, symmetric={symmetric}, channel_wise={channel_wise}, scaler={scaler.__name__}, leaf_param={leaf_param}'
        return s.format(**self.__dict__)


QMODE = Enum('QMODE', ('QDIFF', 'NORMAL', 'PTQD'))


class QuantLayer(nn.Module):

    QMAP = {
        nn.Conv2d: F.conv2d,
        nn.Linear: F.linear,
    }

    def __init__(self,
                 layer: Union[nn.Conv2d, nn.Linear, nn.Conv1d],
                 wq_params: dict = {},
                 aq_params: dict = {}, 
                 disable_aq: bool = False,
                 aq_mode: List[int] = [QMODE.QDIFF.value],
                 quant_emb: bool = False
                 ) -> None:
        super().__init__()
        self.wq_params = wq_params
        self.aq_params = aq_params
        self.fwd_kwargs = {}
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d):
            self.fwd_kwargs = dict(
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups
            )
        self.kwd_func = self.QMAP[type(layer)]
        self.w = layer.weight
        self.original_w = self.w.data.clone()
        self.b = None
        self.original_b = None
        if layer.bias is not None:
            self.b = layer.bias
            self.original_b = self.b.data.clone()
        self.use_wq = False
        self.use_aq = False
        self.disable_aq = disable_aq
        self.aq_mode = aq_mode
        self.quant_emb = quant_emb
        self.wq_params['quant_emb'] = quant_emb
        self.wqtizer = UniformAffineQuantizer(**self.wq_params)
        self.aqtizer = UniformAffineQuantizer(**self.aq_params)
        self.split = 0
        self.act_func = StraightThrough()
        self.ignore_recon = False
        self.extra_repr = layer.extra_repr
    
    def forward(self, 
                x: torch.Tensor,
                split: int = 0
                ) -> torch.Tensor:
        if split != 0 and self.split == 0:
            if '0' in str(x.device):
                logger.info(f'split: {split}')
            self.split = split
            if QMODE.QDIFF.value in self.aq_mode:
                self.aqtizer1 = UniformAffineQuantizer(**self.aq_params)
                self.wqtizer1 = UniformAffineQuantizer(**self.wq_params)
        if self.use_aq and not self.disable_aq:
            if self.split != 0 and QMODE.QDIFF.value in self.aq_mode:
                x1 = self.aqtizer(x[:, :self.split, :, :])
                x2 = self.aqtizer1(x[:, self.split:, :, :])
                x = torch.cat([x1, x2], dim=1)
            else:
                x = self.aqtizer(x)
        if self.use_wq: 
            if self.split != 0 and QMODE.QDIFF.value in self.aq_mode:
                w1 = self.wqtizer(self.w[:, :self.split, :, :])
                w2 = self.wqtizer1(self.w[:, self.split:, :, :])
                w = torch.cat([w1, w2], dim=1)
            else:
                w = self.wqtizer(self.w)
            b = self.b
        else:
            w = self.original_w
            b = self.original_b
        w = w.to(x.device)
        if type(b) == torch.Tensor:
            b = b.to(x.device)
        x = self.kwd_func(x, w, b, **self.fwd_kwargs)
        x = self.act_func(x)
        return x
    
    def set_quant_state(self,
                        use_wq: bool = False,
                        use_aq: bool = False
                        ) -> None:
        self.use_wq = use_wq if not self.ignore_recon else False
        self.use_aq = use_aq if not self.ignore_recon else False


    def set_running_stat(self,
                         running_stat: bool
                         ) -> None:
        self.aqtizer.running_stat = running_stat
        if self.split != 0 and QMODE.QDIFF.value in self.aq_mode:
            self.aqtizer1.running_stat = running_stat