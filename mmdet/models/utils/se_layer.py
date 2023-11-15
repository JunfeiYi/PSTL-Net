# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule


class SELayer(BaseModule):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid')),
                 init_cfg=None):
        super(SELayer, self).__init__(init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out

class DyReLU(BaseModule):
    """Dynamic ReLU (DyReLU) module.
    See `Dynamic ReLU <https://arxiv.org/abs/2003.10027>`_ for details.
    Current implementation is specialized for task-aware attention in DyHead.
    HSigmoid arguments in default act_cfg follow DyHead official code.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py
    Args:
        channels (int): The input (and output) channels of DyReLU module.
        ratio (int): Squeeze ratio in Squeeze-and-Excitation-like module,
            the intermediate channel will be ``int(channels/ratio)``.
            Default: 4.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='HSigmoid', bias=3.0,
            divisor=6.0))
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 channels,
                 ratio=4,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'),
                          dict(type='HSigmoid', bias=3.0, divisor=6.0)),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.channels = channels
        self.expansion = 4  # for a1, b1, a2, b2
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels * self.expansion,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        """Forward function."""
        coeffs = self.global_avgpool(x)
        coeffs = self.conv1(coeffs)
        coeffs = self.conv2(coeffs) - 0.5  # value range: [-0.5, 0.5]
        a1, b1, a2, b2 = torch.split(coeffs, self.channels, dim=1)
        a1 = a1 * 2.0 + 1.0  # [-1.0, 1.0] + 1.0
        a2 = a2 * 2.0  # [-1.0, 1.0]
        out = torch.max(x * a1 + b1, x * a2 + b2)
        return out

class DyReLUv1(BaseModule):
    """Dynamic ReLU (DyReLU) module.
    See `Dynamic ReLU <https://arxiv.org/abs/2003.10027>`_ for details.
    Current implementation is specialized for task-aware attention in DyHead.
    HSigmoid arguments in default act_cfg follow DyHead official code.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py
    Args:
        channels (int): The input (and output) channels of DyReLU module.
        ratio (int): Squeeze ratio in Squeeze-and-Excitation-like module,
            the intermediate channel will be ``int(channels/ratio)``.
            Default: 4.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='HSigmoid', bias=3.0,
            divisor=6.0))
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 channels,
                 out_channels,
                 ratio=4,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'),
                          dict(type='HSigmoid', bias=3.0, divisor=6.0)),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.channels = channels
        self.out_channels = out_channels
        self.num_x = self.channels // self.out_channels
        self.expansion = 2 * self.num_x  # for a1, b1, a2, b2
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=self.out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        """Forward function."""
        coeffs = self.global_avgpool(x) #1024
        coeffs = self.conv1(coeffs)
        coeffs = self.conv2(coeffs) - 0.5  # value range: [-0.5, 0.5]
        a_b = torch.split(coeffs, self.out_channels, dim=1)
        # a1, b1, a2, b2, a3, b3, a4, b4 = torch.split(coeffs, self.channels, dim=1)
        x_s = torch.split(x, self.out_channels, dim=1)
        a_b_list = []
        for i in range(len(a_b)):
            if i % 2 ==0:
                a_b_list.append(a_b[i] * 2.0 + 1.0)
            else:
                a_b_list.append(a_b[i] * 2.0)
        out_x = []
        for i, x_i in enumerate(x_s):
            a_i = 2 * i
            b_i = 2 * i+1
            out_x.append(x_i * a_b_list[a_i] + a_b_list[b_i])
        #a3 = a3 * 2.0 + 1.0  # [-1.0, 1.0] + 1.0
        #a4 = a4 * 2.0  # [-1.0, 1.0]
        out_x = torch.stack(out_x)
        out,_ = torch.max(out_x, dim=0)
        #out = torch.max(x1 * a1 + b1, x2 * a2 + b2, x3 * a3 + b3, x4 * a4 + b4)
        return out