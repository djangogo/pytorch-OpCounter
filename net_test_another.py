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

    def count_flops(self):
        res = 0
        for module in self.children():
            res += module.count_ops()
        res += self.__flops__
        return res


nn.Module = Module


class Conv2d(nn.Conv2d, Module):
    def forward(self, input):
        global multiply_adds
        groups = self.groups
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] \
                     * (self.in_channels / groups) * (1 if multiply_adds else 2)
        bias_ops = 1 if self.bias else 0
        ops_per_element = kernel_ops + bias_ops
        batch_size, input_planes, input_height, input_width = input.size()
        # :math: `H_out} = floor((H_{ in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
        # :math: `W_{out} = floor((W_{ in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
        output_height = math.floor((input_height + 2 * self.padding[0] - self.kernel_size[0] )
                                   / self.stride[0] + 1)
        output_width = math.floor((input_width + 2 * self.padding[1] - self.kernel_size[1] )
                                  / self.stride[1] + 1)
        self.__flops__ = batch_size * self.out_channels * output_width * output_height * ops_per_element
        return super(Conv2d, self).forward(input)


nn.Conv2d = Conv2d


class Linear(nn.Linear, Module):
    def forward(self, input):
        batch_size = input.size(0) if input.dim() == 2 else 1
        weight_ops = self.weight.nelement() * (1 if multiply_adds else 2)
        bias_ops = self.bias.nelement()
        ops_per_sample = weight_ops + bias_ops
        self.__flops__ = batch_size * ops_per_sample
        return super(Linear, self).forward(input)


nn.Linear = Linear


class BatchNorm2d(nn.BatchNorm2d, Module):
    def forward(self, input):
        self.__flops__ = input.nelement()
        return super(BatchNorm2d, self).forward(input)


nn.BatchNorm2d = BatchNorm2d


class ReLU(nn.ReLU, Module):
    def forward(self, input):
        self.__flops__ = input.nelement()
        return super(ReLU, self).forward(input)


nn.ReLU = ReLU


class MaxPool2d(nn.MaxPool2d, Module):
    def forward(self, input):
        batch_size, input_planes, input_height, input_width = input.size()
        self.kernel_size = _pair(self.kernel_size)
        self.dilation = _pair(self.dilation)
        self.padding = _pair(self.padding)
        self.stride = _pair(self.stride)
        kernel_ops = self.kernel_size[0] * self.kernel_size[1]

        # :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1)
        #                           / stride[0] + 1)`
        # :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1)
        #                           / stride[1] + 1)`
        output_height = math.floor((input_height + 2 * self.padding[0] -
                                    self.dilation[0] * (self.kernel_size[0] - 1) - 1)
                                   / self.stride[0] + 1)
        output_width = math.floor((input_width + 2 * self.padding[1] -
                                   self.dilation[1] * (self.kernel_size[1] - 1) - 1)
                                  / self.stride[1] + 1)
        self.__flops__ = batch_size * input_planes * output_width * output_height * kernel_ops
        return super(MaxPool2d, self).forward(input)


nn.MaxPool2d = MaxPool2d


class AvgPool2d(nn.AvgPool2d, Module):
    def forward(self, input):
        batch_size, input_planes, input_height, input_width = input.size()
        self.kernel_size = _pair(self.kernel_size)
        self.padding = _pair(self.padding)
        self.stride = _pair(self.stride)
        kernel_ops = self.kernel_size[0] * self.kernel_size[1]

        # :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - kernel\_size[0])
        #                           / stride[0] + 1)`
        #  :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - kernel\_size[1])
        #                           / stride[1] + 1)`

        output_height = math.floor((input_height + 2 * self.padding[0] - self.kernel_size[0])
                                   / self.stride[0] + 1)
        output_width = math.floor((input_width + 2 * self.padding[1] - self.kernel_size[1])
                                  / self.stride[1] + 1)
        self.__flops__ = batch_size * input_planes * output_width * output_height * kernel_ops
        return super(AvgPool2d, self).forward(input)


nn.AvgPool2d = AvgPool2d


class Sequential(nn.Sequential, Module):
    pass


nn.Sequential = Sequential

from utils import calculate_flops

if __name__ == "__main__":
    # dense-121-32:  7.98  (6, 12, 24, 16)
    # dense-169-32: 14.15  (6, 12, 32, 32)
    # dense-201-32: 20.01  (6, 12, 48, 32)
    # dense-161-48: 28.68  (6, 12, 36, 24)
    import torch
    from torchvision import models
    sample_data = torch.zeros([1, 3, 224, 224])
    sample_var = Variable(sample_data)
    net = models.densenet161(pretrained=False)
    #
    from denselink_imagenet import DenseNet

    net = DenseNet(growthRate=64, depth=-1, nClasses=1000, reduction=0.5, bottleneck=True,
                   #grate_per_stage=(6, 12, 24, 16),
                   layer_per_stage=(6, 12, 24, 16),
                   fetch="exp")

    net(sample_var)
    print('  + Number of FLOPs: %.2fG' % (net.count_ops() / 1e9))

    # print(net)
    total = sum([p.data.nelement() for p in net.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

