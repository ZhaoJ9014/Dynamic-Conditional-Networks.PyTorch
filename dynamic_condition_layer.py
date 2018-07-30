import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

# dynamic conditional conv layer (it is fc layer when the kernel size is 1x1 and the input is cx1x1)
class ConvBasis2d(nn.Module):
    def __init__(self, idfcn, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, transposed=False, output_padding=_pair(0), groups=1, bias=True):
        super(ConvBasis2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.idfcn = idfcn  # the dimension of coditional input
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.weight_basis = Parameter(torch.Tensor(idfcn*out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight_basis.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, input, idw):
        # idw: conditional input
        output = F.conv2d(input, self.weight_basis, self.bias, self.stride, self.padding, self.dilation, self.groups)
        output = output.view(output.size(0), self.idfcn, self.out_channels, output.size(2), output.size(3)) * \
                 idw.view(-1, self.idfcn, 1, 1, 1).expand(output.size(0), self.idfcn, self.out_channels, output.size(2), output.size(3))
        output = output.sum(1).view(output.size(0), output.size(2), output.size(3), output.size(4))
        return output


# an example using dynamic conditional layer
class condition_idfcn_basis_comb_resnet(nn.Module):
  def __init__(self, resnet, fcn):
    super(condition_idfcn_basis_comb_resnet, self).__init__()
    self.resnet = resnet
    self.id_fc = nn.Linear(459558, fcn)
    self.id_tanh = nn.Tanh()
    self.conv_basis = ConvBasis2d(fcn, resnet.fc.in_features, 512, kernel_size=3, padding=1, bias=False)
    self.relu = nn.ReLU(inplace=True)
    self.fc_output = nn.Linear(512, 5)

  def forward(self, x1, x2):
    x1 = self.resnet.conv1(x1)
    x1 = self.resnet.bn1(x1)
    x1 = self.resnet.relu(x1)
    x1 = self.resnet.maxpool(x1)

    x1 = self.resnet.layer1(x1)
    x1 = self.resnet.layer2(x1)
    x1 = self.resnet.layer3(x1)
    x1 = self.resnet.layer4(x1)

    x2 = self.id_fc(x2)
    x2 = self.id_tanh(x2)
    x3 = self.conv_basis(x1, x2)

    x3 = self.relu(x3)
    x3 = self.resnet.avgpool(x3)
    x3 = x3.view(x3.size(0), -1)
    x3 = self.fc_output(x3)

    return x3
