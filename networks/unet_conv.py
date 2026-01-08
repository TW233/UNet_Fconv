import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, sizeP, inNum, outNum, padding=None):
        super(Conv, self).__init__()
        if padding is None:
            padding = sizeP // 2
        self.conv = nn.Conv2d(inNum, outNum, sizeP, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(outNum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv_Down(nn.Module):
    def __init__(self, sizeP, inNum, outNum, padding=None, stride=2):
        super(Conv_Down, self).__init__()
        self.conv_down = nn.Conv2d(inNum, outNum, sizeP, stride=stride, padding=padding, bias=True)
        self.bn_down = nn.BatchNorm2d(outNum)
        self.relu_down = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu_down(x)
        return x


class Conv_Up(nn.Module):
    def __init__(self, sizeP, inNum, outNum, output_padding=None, stride=2):
        super(Conv_Up, self).__init__()
        if output_padding is None:
            output_padding = 0
        self.conv_up = nn.ConvTranspose2d(inNum, outNum, sizeP, stride=stride, output_padding=output_padding, bias=True)
        self.bn_up = nn.BatchNorm2d(outNum)
        self.relu_up = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu_up(x)
        return x


class Conv_Out(nn.Module):
    def __init__(self, sizeP, inNum, outNum, padding=None):
        super(Conv_Out, self).__init__()
        self.conv_out = nn.Conv2d(inNum, outNum, 1)
        self.bn_out = nn.BatchNorm2d(outNum)
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_out(x)
        x = self.bn_out(x)
        x = self.relu_out(x)
        return x


class Inconv(nn.Module):
    def __init__(self, sizeP, inNum, outNum, padding=None):
        super(Inconv, self).__init__()
        self.conv1 = Conv(sizeP, inNum, outNum, padding)
        self.conv2 = Conv(sizeP, outNum, outNum, padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Down(nn.Module):
    def __init__(self, sizeP, inNum, outNum, padding=None, stride=2):
        super(Down, self).__init__()
        self.conv1 = Conv_Down(sizeP, inNum, outNum, padding, stride)
        self.conv2 = Conv(sizeP, outNum, outNum, padding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Up(nn.Module):
    def __init__(self, sizeP, inNum, outNum, output_padding=None, stride=2):
        super(Up, self).__init__()
        self.up = Conv_Up(sizeP, inNum // 2, inNum // 2, output_padding, stride)
        self.conv3 = Conv(sizeP, inNum, outNum)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv3(x)
        return x


class finalconv(nn.Module):
    def __init__(self, inNum, outNum):
        super(finalconv, self).__init__()
        self.finalconv = nn.Conv2d(inNum, outNum, 1)
        self.bn = nn.BatchNorm2d(outNum)

    def forward(self, x):
        x = self.finalconv(x)
        x = self.bn(x)
        return x


class unet_conv(nn.Module):
    def __init__(self, sizeP=5, in_channels=3, classes=2):
        super(unet_conv, self).__init__()

        self.inc = Inconv(7, in_channels, 64)
        self.down1 = Down(sizeP, 64, 128, 2, 2)
        self.down2 = Down(sizeP, 128, 256, 2, 2)
        self.down3 = Down(sizeP, 256, 512, 2, 2)
        self.down4 = Down(sizeP, 512, 512, 2, 2)
        self.up1 = Up(sizeP, 1024, 256, None, 2)
        self.up2 = Up(sizeP, 512, 128, None, 2)
        self.up3 = Up(sizeP, 256, 64, None, 2)
        self.up4 = Up(sizeP, 128, 64, None, 2)
        self.outc = finalconv(64, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        return x


if __name__ == '__main__':
    device = torch.device('cuda:0')
    img = torch.rand(2, 3, 448, 448)
    # print(img)
    net = unet_conv()
    for name, param in net.named_parameters():
        nn.init.normal_(param, -0.01, 0.02)
    output = net(img)
    # print(output)
    img1 = torch.rot90(img, dims=(-1, -2))
    output1 = net(img1)
    output1 = torch.rot90(output1, k=-1, dims=(-1, -2))
    difference = output1 - output
    print(difference)