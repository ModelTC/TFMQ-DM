import torch
from typing import Dict, Union
import torch.nn.functional as F
import torch.nn as nn
from quant.quant_layer import QuantLayer
from quant.quant_model import QuantModel
from quant.quant_block import BaseQuantBlock, QuantBasicTransformerBlock
from typing import Union, Tuple
import logging
logger = logging.getLogger(__name__)


def save_inout(model: QuantModel, 
               layer: Union[QuantLayer, BaseQuantBlock], 
               cali_data: Tuple[torch.Tensor], 
               asym: bool = False, 
               use_act: bool = False, 
               batch_size: int = 128, 
               keep_gpu: bool = True, 
               ) -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
    device = next(model.parameters()).device
    get_inout = GetLayerInpOut(model, layer, device, asym, use_act)
    cached_inputs, cached_outputs = None, None
    torch.cuda.empty_cache()
    
    for i in range(0, cali_data[0].size(0), batch_size):
        ipts, opts = get_inout(*(_[i: i + batch_size] for _ in cali_data))
        if cached_inputs is None:
            cached_inputs = tuple([] for _ in range(len(ipts)))
        for j in range(len(ipts)):
            cached_inputs[j].append(ipts[j].cpu())
        if cached_outputs is None:
            cached_outputs = tuple([] for _ in range(len(opts)))
        for j in range(len(opts)):
            cached_outputs[j].append(opts[j].cpu())
    cached_inputs = tuple((torch.cat([y for y in x]) for x in cached_inputs))
    cached_outputs = tuple((torch.cat([y for y in x]) for x in cached_outputs))
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_inputs = tuple(x.to(device) for x in cached_inputs)
        cached_outputs = tuple(x.to(device) for x in cached_outputs)
    # assert len(cached_outputs) == 1
    for i, x in enumerate(cached_inputs):
        if '0' in str(x.device) or 'cpu' in str(x.device):
            logger.info(f'input {i} shape: {x.shape}')
    for i, x in enumerate(cached_outputs):
        if '0' in str(x.device) or 'cpu' in str(x.device):
            logger.info(f'output {i} shape: {x.shape}')
    if len(cached_outputs) == 1:
        return cached_inputs, cached_outputs[0]
    return cached_inputs, cached_outputs  


def save_grad(model: QuantModel, 
              layer: Union[QuantLayer, BaseQuantBlock], 
              cali_data: Tuple[torch.Tensor], 
              damping: float = 1., 
              use_aq: bool = False, 
              batch_size: int = 32, 
              keep_gpu: bool = True
              ) -> torch.Tensor:
    device = next(model.parameters()).device
    get_grad = GetLayerGrad(model, layer, device, use_aq)
    cached_grads = []
    torch.cuda.empty_cache()
    for i in range(0, cali_data[0].size(0), batch_size):
        cached_grads.append(get_grad(*(_[i: i + batch_size] for _ in cali_data)).cpu())
    cached_grads = torch.cat([x for x in cached_grads])
    cached_grads = cached_grads.abs() + 1.0
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_grads = cached_grads.to(device)
    return cached_grads


class StopForwardException(Exception):
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, 
                 store_input: bool = False, 
                 store_output: bool = False, 
                 stop_forward: bool = False
                 ) -> None:
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, 
                 module: nn.Module, 
                 input_batch: Tuple[Union[torch.Tensor, int]], # ResBlock or ResnetBlock has an integer input split
                 kwargs: Dict[str, Union[torch.Tensor, int]],
                 output_batch: Union[torch.Tensor, Tuple[torch.Tensor]]
                 ) -> None:
        if self.store_input:
            self.input_store = input_batch
            if isinstance(input_batch[-1], int):
                self.input_store = input_batch[:-1]
            if isinstance(module, QuantBasicTransformerBlock): # in order to capture context
                self.input_store = (input_batch[0], kwargs["context"])
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    def __init__(self, model: QuantModel, 
                 layer: Union[QuantLayer, BaseQuantBlock],
                 device: torch.device, 
                 asym: bool = False, 
                 use_aq: bool = False
                 ) -> None:
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.use_aq = use_aq
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, 
                 xs: torch.Tensor, 
                 ts: torch.Tensor, 
                 cs: torch.Tensor = None
                 ) -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
        self.model.eval()
        self.model.set_quant_state(False, False)

        handle = self.layer.register_forward_hook(self.data_saver, with_kwargs=True)
        with torch.no_grad():
            try:
                if cs is not None:
                    _ = self.model(xs.to(self.device), ts.to(self.device), cs.to(self.device))
                else:
                    _ = self.model(xs.to(self.device), ts.to(self.device))
            except StopForwardException:
                pass

            if self.asym:
                # Recalculate input with network quantized
                self.data_saver.store_output = False
                self.model.set_quant_state(use_wq=True, use_aq=self.use_aq)
                try:
                    if cs is not None:
                        _ = self.model(xs.to(self.device), ts.to(self.device), cs.to(self.device))
                    else:
                        _ = self.model(xs.to(self.device), ts.to(self.device))
                except StopForwardException:
                    pass
                self.data_saver.store_output = True

        handle.remove()
        
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.use_aq)
        self.model.train()
        input_stores = tuple(x.detach() for x in self.data_saver.input_store)
        if isinstance(self.data_saver.output_store, torch.Tensor):
            output_stores = tuple([self.data_saver.output_store.detach()])
            return input_stores, output_stores
        output_stores = tuple(x.detach() for x in self.data_saver.output_store)
        return input_stores, output_stores


class GradSaverHook:
    def __init__(self, 
                 store_grad: bool = True
                 ) -> None:
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, 
                 module: nn.Module, 
                 grad_input: torch.Tensor, 
                 grad_output: torch.Tensor
                 ) -> None:
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetLayerGrad:
    def __init__(self, 
                 model: QuantModel, 
                 layer: Union[QuantLayer, BaseQuantBlock],
                 device: torch.device, 
                 use_aq: bool = False
                 ) -> None:
        self.model = model
        self.layer = layer
        self.device = device
        self.use_aq = use_aq
        self.data_saver = GradSaverHook(True)

    def __call__(self, 
                 xs: torch.Tensor, 
                 ts: torch.Tensor, 
                 cs: torch.Tensor = None
                 ) -> torch.Tensor:
        """
        Compute the gradients of block output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model

        :param model_input: calibration data samples
        :return: gradients
        """
        def quantize_model_till(model: QuantModel,
                                layer: Union[QuantLayer, BaseQuantBlock],
                                use_aq: bool = False
                                ) -> None:
            """
            We assumes modules are correctly ordered, holds for all models considered
            :param model: quantized_model
            :param layer: a block or a single layer.
            """
            model.set_quant_state(False, False)
            for _, module in model.named_modules():
                if isinstance(module, (QuantLayer, BaseQuantBlock)):
                    module.set_quant_state(True, use_aq)
                if module == layer:
                    break
                
        self.model.eval()
        handle = self.layer.register_backward_hook(self.data_saver)
        with torch.enable_grad():
            try:
                self.model.zero_grad()
                self.model.set_quant_state(False, False)
                if cs is not None:
                    out_fp = self.model(xs.to(self.device), ts.to(self.device), cs.to(self.device))
                else:
                    out_fp = self.model(xs.to(self.device), ts.to(self.device))
                quantize_model_till(self.model, self.layer, self.use_aq)
                if cs is not None:
                    out_q = self.model(xs.to(self.device), ts.to(self.device), cs.to(self.device))
                else:
                    out_q = self.model(xs.to(self.device), ts.to(self.device))
                loss = F.kl_div(F.log_softmax(out_q, dim=1), F.softmax(out_fp, dim=1), reduction='batchmean')
                loss.backward()
            except StopForwardException:
                pass

        handle.remove()
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.use_aq)
        self.model.train()
        return self.data_saver.grad_out.data








            