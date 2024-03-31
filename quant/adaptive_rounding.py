import torch 
from torch import nn
from quant.quant_layer import UniformAffineQuantizer, ste_round
from enum import Enum

RMODE = Enum('RMODE', ('LEARNED_ROUND_SIGMOID', 
                       'NEAREST', 
                       'NEAREST_STE', 
                       'STOCHASTIC',
                       'LEARNED_HARD_SIGMOID'))

class AdaRoundQuantizer(nn.Module):

    def __init__(self,
                 uaqtizer: UniformAffineQuantizer,
                 w: torch.Tensor,
                 rmode: RMODE = RMODE.LEARNED_ROUND_SIGMOID,
                ) -> None:
        
        super().__init__()
        self.level = uaqtizer.level
        self.symmetric = uaqtizer.symmetric
        self.delta = uaqtizer.delta
        self.zero_point = uaqtizer.zero_point
        self.rmode = rmode
        self.soft_tgt = False 
        self.gamma, self.zeta = -0.1, 1.1
        self.alpha = None 
        self.init_alpha(x=w.clone())

    def init_alpha(self, x: torch.Tensor) -> None:
        self.delta = self.delta.to(x.device)
        if self.rmode == RMODE.LEARNED_HARD_SIGMOID:
            rest = (x / self.delta) - torch.floor(x / self.delta)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError
        
    def get_soft_tgt(self) -> torch.Tensor:
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        if isinstance(self.delta, torch.Tensor):
            self.delta = self.delta.to(x.device)
        if isinstance(self.zero_point, torch.Tensor):
            self.zero_point = self.zero_point.to(x.device)

        x_floor = torch.floor(x / self.delta)
        if self.rmode == RMODE.NEAREST:
            x_int  = torch.round(x / self.delta)
        elif self.rmode == RMODE.NEAREST_STE:
            x_int = ste_round(x / self.delta)
        elif self.rmode == RMODE.STOCHASTIC:
            x_int = x_floor + torch.bernoulli((x / self.delta) - x_floor)
        elif self.rmode == RMODE.LEARNED_HARD_SIGMOID:
            if self.soft_tgt:
                x_int = x_floor + self.get_soft_tgt().to(x.device)
            else:
                self.alpha = self.alpha.to(x.device)
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise NotImplementedError
        
        NB, PB = -self.level // 2 if self.symmetric else 0, self.level // 2 - 1 if self.symmetric else self.level - 1
        x_q = torch.clamp(x_int + self.zero_point, NB, PB)
        x_dq = self.delta * (x_q - self.zero_point)
        return x_dq
    
    def extra_repr(self) -> str:
        s = 'level={}, symmetric={}, rmode={}'.format(self.level, self.symmetric, self.rmode)
        return s.format(**self.__dict__)


                 