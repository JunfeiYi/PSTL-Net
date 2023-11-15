import math

import torch
from mmcv.utils import ext_loader
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair, _single
from torch.autograd.function import once_differentiable

ext_module = ext_loader.load_ext('_ext', [
    'deform_conv_forward', 'deform_conv_backward_input',
    'deform_conv_backward_parameters'
])


class DeformConvFunction(Function):

    @staticmethod
    def forward(ctx,
                input,
                offset,
                weight,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                deformable_groups=1,
                bias=False,
                im2col_step=64):
        if input is not None and input.dim() != 4:
            raise ValueError(
                'Expected 4D tensor as input, got {}D tensor instead.'.format(
                    input.dim()))
        assert bias is False, 'Only support bias is False.'
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step

        # input = input.type_as(offset)
        # weight = weight.type_as(input)
        ctx.save_for_backward(input, offset, weight)

        output = input.new_empty(
            DeformConvFunction._output_size(input, weight, ctx.padding,
                                            ctx.dilation, ctx.stride))

        ctx.bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones

        cur_im2col_step = min(ctx.im2col_step, input.shape[0])
        assert (input.shape[0] % cur_im2col_step) == 0, 'im2col step must divide batchsize'
        ext_module.deform_conv_forward(
            input, weight, offset, output, ctx.bufs_[0], ctx.bufs_[1],
            weight.size(3), weight.size(2), ctx.stride[1], ctx.stride[0],
            ctx.padding[1], ctx.padding[0], ctx.dilation[1],
            ctx.dilation[0], ctx.groups, ctx.deformable_groups,
            cur_im2col_step)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, weight = ctx.saved_tensors

        grad_input = grad_offset = grad_weight = None

        cur_im2col_step = min(ctx.im2col_step, input.shape[0])
        assert (input.shape[0] % cur_im2col_step) == 0, 'im2col step must divide batchsize'

        grad_output = grad_output.contiguous()
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_input = torch.zeros_like(input)
            grad_offset = torch.zeros_like(offset)
            ext_module.deform_conv_backward_input(
                input, offset, grad_output, grad_input,
                grad_offset, weight, ctx.bufs_[0], weight.size(3),
                weight.size(2), ctx.stride[1], ctx.stride[0],
                ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                ctx.dilation[0], ctx.groups, ctx.deformable_groups,
                cur_im2col_step)

        if ctx.needs_input_grad[2]:
            grad_weight = torch.zeros_like(weight)
            ext_module.deform_conv_backward_parameters(
                input, offset, grad_output,
                grad_weight, ctx.bufs_[0], ctx.bufs_[1], weight.size(3),
                weight.size(2), ctx.stride[1], ctx.stride[0],
                ctx.padding[1], ctx.padding[0], ctx.dilation[1],
                ctx.dilation[0], ctx.groups, ctx.deformable_groups, 1,
                cur_im2col_step)

        return (grad_input, grad_offset, grad_weight, None, None, None, None,
                None)

    @staticmethod
    def _output_size(input, weight, padding, dilation, stride):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be {})'.format(
                    'x'.join(map(str, output_size))))
        return output_size


deform_conv = DeformConvFunction.apply


class DeformConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deformable_groups=1,
                 bias=False):
        super(DeformConv, self).__init__()

        assert not bias
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, offset):
        # To fix an assert error in deform_conv_cuda.cpp:128
        # input image is smaller than kernel
        input_pad = (
            x.size(2) < self.kernel_size[0] or x.size(3) < self.kernel_size[1])
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant',
                           0).contiguous()
        out = deform_conv(x, offset, self.weight, self.stride, self.padding,
                          self.dilation, self.groups, self.deformable_groups)
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) -
                      pad_w].contiguous()
        return out


class sepc_conv(DeformConv):
    def __init__(self, *args, part_deform=False, **kwargs):
        super(sepc_conv, self).__init__(*args, **kwargs)
        self.part_deform = part_deform
        if self.part_deform:
            self.conv_offset = nn.Conv2d(self.in_channels,
                                         self.deformable_groups * 2 *
                                         self.kernel_size[0] *
                                         self.kernel_size[1],
                                         kernel_size=self.kernel_size,
                                         stride=_pair(self.stride),
                                         padding=_pair(self.padding),
                                         bias=True)
            self.init_offset()

        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        self.start_level = 1

    def init_offset(self):

        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, i, x):
        if i < self.start_level or not self.part_deform:
            return torch.nn.functional.conv2d(x,
                                              self.weight,
                                              bias=self.bias,
                                              stride=self.stride,
                                              padding=self.padding,
                                              dilation=self.dilation,
                                              groups=self.groups)

        offset = self.conv_offset(x)
        return deform_conv(x, offset, self.weight, self.stride, self.padding,
                           self.dilation, self.groups,
                           self.deformable_groups) + self.bias.unsqueeze(
                               0).unsqueeze(-1).unsqueeze(-1)




