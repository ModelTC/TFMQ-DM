import torch
from enum import Enum
from quant.quant_block import BaseQuantBlock, QuantResBlock, QuantTemporalInformationBlock, QuantTemporalInformationBlockDDIM
from quant.quant_layer import QMODE, QuantLayer, lp_loss
from quant.adaptive_rounding import AdaRoundQuantizer
from typing import Union
import logging
logger = logging.getLogger(__name__)

RLOSS = Enum('RLOSS', ('RELAXATION', 'MSE', 'FISHER_DIAG', 'FISHER_FULL', 'NONE'))
print_freq = 2000

class LossFunc:
    def __init__(self,
                 o: Union[QuantLayer, BaseQuantBlock],
                 round_loss: RLOSS = RLOSS.RELAXATION,
                 w: float = 1.,
                 rec_loss: RLOSS = RLOSS.MSE,
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.) -> None:
        self.o = o
        self.round_loss = round_loss
        self.w = w
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p 
        self.temp_decay = LinearTempDecay(t_max = max_count, 
                                          rel_start_decay = warmup + (1 - warmup) * decay_start,
                                          start_b = b_range[0], 
                                          end_b = b_range[1])
        self.count = 0

    def __call__(self, 
                 pred: torch.Tensor, 
                 tgt: torch.Tensor, 
                 grad: torch.Tensor = None) -> torch.Tensor:
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == RLOSS.MSE:
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == RLOSS.FISHER_DIAG:
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == RLOSS.FISHER_FULL:
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == RLOSS.NONE:
            b = round_loss = 0
        elif self.round_loss == RLOSS.RELAXATION:
            if isinstance(self.o, QuantLayer):
                round_vals: torch.Tensor = self.o.wqtizer.get_soft_tgt()
                round_loss = self.w * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            else:
                self.o: BaseQuantBlock
                round_loss = 0
                for _, module in self.o.named_modules():
                    if isinstance(module, QuantLayer) and module.quant_emb is False:
                        if not module.ignore_recon:
                            if module.split == 0 or QMODE.QDIFF.value not in module.aq_mode:
                                round_vals: torch.Tensor = module.wqtizer.get_soft_tgt()
                                round_loss += self.w * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
                            else:
                                round_vals: torch.Tensor = module.wqtizer.get_soft_tgt()
                                round_vals1: torch.Tensor = module.wqtizer1.get_soft_tgt()
                                round_loss += self.w * ((1 - ((round_vals - .5).abs() * 2).pow(b)).sum() * module.split \
                                    + (1 - ((round_vals1 - .5).abs() * 2).pow(b)).sum() * (module.w.shape[1] - module.split)) / module.w.shape[1]
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % print_freq == 0:
            logger.info('Total loss:\t{:.8f} (rec:{:.8f}, round:{:.8f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


class LossFuncTimeEmbedding:
    def __init__(self,
                 o: Union[QuantLayer, BaseQuantBlock],
                 round_loss: RLOSS = RLOSS.RELAXATION,
                 w: float = 1.,
                 rec_loss: RLOSS = RLOSS.MSE,
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.) -> None:
        self.o = o
        self.round_loss = round_loss
        self.w = w
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p
        self.temp_decay = LinearTempDecay(t_max = max_count,
                                          rel_start_decay = warmup + (1 - warmup) * decay_start,
                                          start_b = b_range[0],
                                          end_b = b_range[1])
        self.count = 0

    def __call__(self,
                 preds: torch.Tensor,
                 tgts: torch.Tensor,
                ) -> torch.Tensor:
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        rec_loss = torch.Tensor([0]).to(preds[0].device)
        for pred, tgt in zip(preds, tgts):
            if self.rec_loss == RLOSS.MSE:
                rec_loss += lp_loss(pred, tgt, p=self.p)
            else:
                raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == RLOSS.NONE:
            b = round_loss = 0
        elif self.round_loss == RLOSS.RELAXATION:
            assert isinstance(self.o, (QuantTemporalInformationBlock, QuantTemporalInformationBlockDDIM))
            round_loss = 0
            for _, module in self.o.named_modules():
                if isinstance(module, QuantLayer):
                    if not module.ignore_recon:
                        assert isinstance(module.wqtizer, AdaRoundQuantizer)
                        round_vals = module.wqtizer.get_soft_tgt()
                        round_loss += self.w * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            if isinstance(self.o, QuantTemporalInformationBlock):
                for emb_layers in self.o.emb_layers:
                    for _, module in emb_layers.named_modules():
                        if isinstance(module, QuantLayer):
                            if not module.ignore_recon:
                                assert isinstance(module.wqtizer, AdaRoundQuantizer)
                                round_vals = module.wqtizer.get_soft_tgt()
                                round_loss += self.w * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            else:
                for temb_proj in self.o.temb_projs:
                    assert isinstance(temb_proj, QuantLayer)
                    if not temb_proj.ignore_recon:
                        assert isinstance(temb_proj.wqtizer, AdaRoundQuantizer)
                        round_vals = temb_proj.wqtizer.get_soft_tgt()
                        round_loss += self.w * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % print_freq == 0 and "0" in str(pred.device):
            logger.info('Total loss:\t{:.8f} (rec:{:.8f}, round:{:.8f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, 
                 t_max: int, 
                 rel_start_decay: float = 0.2, 
                 start_b: int = 10, 
                 end_b: int = 2
                 ) -> None:
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t) -> float:
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
