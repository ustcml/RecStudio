import threading
import torch
from torch.cuda._utils import _get_device_index
from torch.cuda.amp import autocast
from torch._utils import ExceptionWrapper

from itertools import chain
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.replicate import replicate
from torch._utils import (
    _get_all_device_indices,
    _get_available_device_type,
    _get_device_index,
)

__all__ = ['data_parallel']


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, (list, tuple)):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply(modules, method_name, inputs, kwargs_tup=None, devices=None):
    r"""Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        method_name (str): the name of method to be called in data parallel
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    assert len(modules) == len(inputs), f'The number of modules {len(modules)} is not equal to the number of inputs {len(inputs)}'
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = [_get_device_index(x, True) for x in devices]
    streams = [torch.cuda.current_stream(x) for x in devices]
    lock = threading.Lock()
    results = {}
    grad_enabled, autocast_enabled = torch.is_grad_enabled(), torch.is_autocast_enabled()

    def _worker(i, module, input, kwargs, device=None, stream=None):
        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        if stream is None:
            stream = torch.cuda.current_stream(device)
        try:
            with torch.cuda.device(device), torch.cuda.stream(stream), autocast(enabled=autocast_enabled):
                # this also avoids accidental slicing of `input` if it is a Tensor
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = getattr(module, method_name)(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where="in replica {} on device {}".format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, kwargs, device, stream))
                   for i, (module, input, kwargs, device, stream) in
                   enumerate(zip(modules, inputs, kwargs_tup, devices, streams))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0], streams[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs


def data_parallel(module, method_name, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.
    This is the functional version of the DataParallel module.
    Args:
        module (Module): the module to evaluate in parallel
        method_name (str): the name of method to be called in data parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,) if inputs is not None else ()

    device_type = _get_available_device_type()

    if device_ids is None:
        device_ids = _get_all_device_indices()

    if output_device is None:
        output_device = device_ids[0]

    device_ids = [_get_device_index(x, True) for x in device_ids]
    output_device = _get_device_index(output_device, True)
    src_device_obj = torch.device(device_type, device_ids[0])

    for t in chain(module.parameters(), module.buffers()):
        if t.device != src_device_obj:
            raise RuntimeError("module must have its parameters and buffers "
                               "on device {} (device_ids[0]) but found one of "
                               "them on device: {}".format(src_device_obj, t.device))

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    # for module without any inputs, empty list and dict will be created
    # so the module can be executed on one device which is the first one in device_ids
    if not inputs and not module_kwargs:
        inputs = ((),)
        module_kwargs = ({},)

    if len(device_ids) == 1:
        return getattr(module, method_name)(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, method_name, inputs, module_kwargs, used_device_ids)
    if outputs[0].dim() == 0:   # for loss output by training step
        outputs = [_.view(-1) for _ in outputs]
        res = gather(outputs, output_device, dim)
        res = res.sum()    # TODO: The `sum` operation would be not right for loss with mean reduction.
    else:
        res = gather(outputs, output_device, dim)
    return res