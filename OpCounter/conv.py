from .module import *

import torch.nn as nn


class Conv2d(nn.Conv2d, Module):
    def count_ops(self):
        global multiply_adds
        groups = self.groups
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] \
                     * (self.in_channels / groups) * (1 if multiply_adds else 2)
        bias_ops = 1 if self.bias else 0
        ops_per_element = kernel_ops + bias_ops
        batch_size, input_planes, input_height, input_width = input.size()
        # :math: `H_out} = floor((H_{ in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
        # :math: `W_{out} = floor((W_{ in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
        output_height = math.floor((input_height + 2 * self.padding[0] - self.kernel_size[0])
                                   / self.stride[0] + 1)
        output_width = math.floor((input_width + 2 * self.padding[1] - self.kernel_size[1])
                                  / self.stride[1] + 1)
        self.__flops__ = batch_size * self.out_channels * output_width * output_height * ops_per_element
        return self.__flops__

    def forward(self, input):
        return super(Conv2d, self).forward(input)


nn.Conv2d = Conv2d
