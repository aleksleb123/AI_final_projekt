from torch import nn
import torch


class SepConv3d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False
    ):
        super(SepConv3d, self).__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            groups=out_channels,
            padding=padding,
            bias=bias,
            stride=stride,
        )

    def forward(self, x):
        x = self.depthwise(x)
        return x


class SplitConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, split_dim, drop_rate):
        super(SplitConvBlock, self).__init__()

        self.split_dim = split_dim

        self.leftconv_1 = SepConvBlock(
            in_channels // 2, mid_channels // 2, (3, 4, 3), droprate=drop_rate
        )
        self.rightconv_1 = SepConvBlock(
            in_channels // 2, mid_channels // 2, (3, 4, 3), droprate=drop_rate
        )

        self.leftconv_2 = SepConvBlock(
            mid_channels // 2, out_channels // 2, (3, 4, 3), droprate=drop_rate
        )
        self.rightconv_2 = SepConvBlock(
            mid_channels // 2, out_channels // 2, (3, 4, 3), droprate=drop_rate
        )

    def forward(self, x):
        (left, right) = torch.tensor_split(x, 2, dim=self.split_dim)

        self.leftblock = nn.Sequential(self.leftconv_1, self.leftconv_2)
        self.rightblock = nn.Sequential(self.rightconv_1, self.rightconv_2)

        left = self.leftblock(left)
        right = self.rightblock(right)
        x = torch.cat((left, right), dim=self.split_dim)
        return x


class MidFlowBlock(nn.Module):
    def __init__(self, channels, drop_rate):
        super(MidFlowBlock, self).__init__()

        self.conv1 = ConvBlock(
            channels, channels, (3, 3, 3), droprate=drop_rate, padding="same"
        )
        self.conv2 = ConvBlock(
            channels, channels, (3, 3, 3), droprate=drop_rate, padding="same"
        )
        self.conv3 = ConvBlock(
            channels, channels, (3, 3, 3), droprate=drop_rate, padding="same"
        )

        # self.block = nn.Sequential(self.conv1, self.conv2, self.conv3)
        self.block = self.conv1

    def forward(self, x):
        x = nn.ELU()(self.block(x) + x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1, 1),
        padding="valid",
        droprate=None,
        pool=False,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm3d(out_channels)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(droprate)

        if pool:
            self.maxpool = nn.MaxPool3d(3, stride=2)
        else:
            self.maxpool = None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.elu(x)

        if self.maxpool:
            x = self.maxpool(x)

        x = self.dropout(x)

        return x


class FullConnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, droprate=0.0):
        super(FullConnBlock, self).__init__()
        self.dense = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(droprate)

    def forward(self, x):
        x = self.dense(x)
        x = self.norm(x)
        x = self.elu(x)
        x = self.dropout(x)
        return x


class SepConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=(1, 1, 1),
        padding="valid",
        droprate=None,
        pool=False,
    ):
        super(SepConvBlock, self).__init__()
        self.conv = SepConv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm3d(out_channels)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(droprate)

        if pool:
            self.maxpool = nn.MaxPool3d(3, stride=2)
        else:
            self.maxpool = None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.elu(x)

        if self.maxpool:
            x = self.maxpool(x)

        x = self.dropout(x)

        return x
