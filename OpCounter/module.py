import math

from torch.nn.modules.conv import _pair
from torch.autograd import Variable
import functools
import torch.nn as nn

multiply_adds = False

class Module(nn.Module):
    __flops__ = 0

    def __call__(self, *input, **kwargs):
        self.__flops__ = 0
        for hook in self._forward_pre_hooks.values():
            hook(self, input)
        result = self.forward(*input, **kwargs)
        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                raise RuntimeError(
                    "forward hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))
        if len(self._backward_hooks) > 0:
            var = result
            while not isinstance(var, Variable):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, Variable)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
        return result

    def count_ops(self):
        res = 0
        for module in self.children():
            res += module.count_ops()
        res += self.__flops__
        return res



nn.Module = Module