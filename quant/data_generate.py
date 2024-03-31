from contextlib import nullcontext
from typing import List, Union
from ddim.models.diffusion import Model
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.dpm_solver.sampler import DPMSolverSampler
from ldm.models.diffusion.plms import PLMSSampler
from typing import Tuple
from torch import autocast
import torch


def generate_cali_text_guided_data(model: LatentDiffusion,
                                   sampler: Union[DPMSolverSampler, PLMSSampler, DDIMSampler],
                                   T: int,
                                   c: int,
                                   batch_size: int,
                                   prompts: Tuple[str],
                                   shape: List[int],
                                   precision_scope: Union[autocast, nullcontext],
                                   ) -> Tuple[torch.Tensor]:
    tmp = list()
    model.eval()
    with torch.no_grad():
        with precision_scope("cuda"):
            for t in range(1, T + 1):
                # x_{t + 1} = f(x_t, t_t, c_t)
                if t % c == 0:
                    for p in prompts:
                        uc_t = model.get_learned_conditioning(batch_size * [""])
                        c_t = model.get_learned_conditioning(batch_size * [p])
                        x_t, t_t = sampler.sample(S=T,
                                                conditioning=c_t,
                                                batch_size=batch_size,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=7.5,
                                                unconditional_conditioning=uc_t,
                                                untill_fake_t=t)
                        if isinstance(sampler, (PLMSSampler, DDIMSampler)):
                            ddpm_time_num = 1000 # in yaml
                            real_time = (T - t) * ddpm_time_num // T + 1
                            t_t = torch.full((batch_size,), real_time, device=sampler.model.betas.device, dtype=torch.long)
                        tmp += [[x_t, t_t, c_t], [x_t, t_t, uc_t]]

    cali_data = ()
    for i in range(len(tmp[0])):
        cali_data += (torch.cat([x[i] for x in tmp]), )
    return cali_data


def generate_cali_data_ddim(runnr,
                       model: Model,
                       T: int,
                       c: int,
                       batch_size: int,
                       shape: List[int],
                       ) -> Tuple[torch.Tensor]:
    tmp = list()
    for i in range(1, T + 1):
        # x_{t + 1} = f(x_t, t_t, c_t)
        if i % c == 0:
            from ddim.runners.diffusion import Diffusion
            runnr: Diffusion
            N, C, H, W = batch_size, *shape
            x = torch.randn((N, C, H, W), device=runnr.device)
            x_t, t_t = runnr.sample_image(x, model, untill_fake_t=i)[1:]
            tmp += [[x_t, t_t]]
    cali_data = ()
    for i in range(len(tmp[0])):
        cali_data += (torch.cat([x[i] for x in tmp]), )
    return cali_data


def generate_cali_data_ldm(model: LatentDiffusion,
                       T: int,
                       c: int,
                       batch_size: int,
                       shape: List[int],
                       vanilla: bool = False,
                       dpm: bool = False,
                       plms: bool = False,
                       eta: float = 0.0,
                       ) -> Tuple[torch.Tensor]:
    if vanilla:
        pass
    elif dpm:
        sampler = DPMSolverSampler(model)
    elif plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    tmp = list()
    for t in range(1, T + 1):
        if t % c == 0:
            if not vanilla:
                x_t, t_t = sampler.sample(S=T,
                                          batch_size=batch_size,
                                          shape=shape,
                                          verbose=False,
                                          eta=eta,
                                          untill_fake_t=t)
                if isinstance(sampler, (PLMSSampler, DDIMSampler)):
                    ddpm_time_num = 1000 # in yaml
                    real_time = (T - t) * ddpm_time_num // T + 1
                    t_t = torch.full((batch_size,), real_time, device=sampler.model.betas.device, dtype=torch.long)
                tmp += [[x_t, t_t]]
            else:
                raise NotImplementedError("Vanilla LDM is not implemented yet, because it needs 1000 steps to generate one sample.")
    cali_data = ()
    for i in range(len(tmp[0])):
        cali_data += (torch.cat([x[i] for x in tmp]), )
    return cali_data


def generate_cali_data_ldm_imagenet(model: LatentDiffusion,
                                    T: int,
                                    c: int,
                                    batch_size: int, # 8
                                    shape: List[int],
                                    eta: float = 0.0,
                                    scale: float = 3.0
                                    ) -> Tuple[torch.Tensor]:
    sampler = DDIMSampler(model)
    tmp = list()
    classes = [i for i in range(0, 1000, 1000 // 31)]
    with  torch.no_grad():
        with model.ema_scope():
            for i in range(1, T + 1):
                if i % c == 0:
                    uc_t = model.get_learned_conditioning(
                        {model.cond_stage_key: torch.tensor(batch_size * [1000]).to(model.device)}
                    )
                    for class_label in classes:
                        xc = torch.tensor(batch_size * [class_label])
                        c_t = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                        x_t, t_t = sampler.sample(S=T,
                                                  batch_size=batch_size,
                                                  shape=shape,
                                                  verbose=False,
                                                  eta=eta,
                                                  unconditional_guidance_scale=scale,
                                                  unconditional_conditioning=uc_t,
                                                  conditioning=c_t,
                                                  untill_fake_t=i)
                        if isinstance(sampler, DDIMSampler):
                            ddpm_time_num = 1000
                            real_time = (T - i) * ddpm_time_num // T + 1
                            t_t = torch.full((batch_size,), real_time, device=sampler.model.betas.device, dtype=torch.long)
                        tmp += [[x_t, t_t, c_t], [x_t, t_t, uc_t]]
    cali_data = ()
    for i in range(len(tmp[0])):
        cali_data += (torch.cat([x[i] for x in tmp]), )
    return cali_data