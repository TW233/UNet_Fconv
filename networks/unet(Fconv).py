import torch
import torch.nn as nn
import torch.nn.functional as F
import networks.F_Conv as fn


class Fconv(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, padding=None, ifIni=0):
        super(Fconv, self).__init__()
        if padding is None:
            padding = sizeP // 2
        self.conv = fn.Fconv_PCA(sizeP, inNum, outNum, tranNum, inP, padding, ifIni, bias=True)
        self.bn = fn.F_BN(outNum, tranNum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Fconv_Down(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, padding=None, stride=2):
        super(Fconv_Down, self).__init__()
        self.conv_down = fn.Fconv_Down(sizeP, inNum, outNum, tranNum, inP, padding, ifIni=0, bias=True, stride=stride)
        self.bn_down = fn.F_BN(outNum, tranNum)
        self.relu_down = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu_down(x)
        return x


class Fconv_Up(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, output_padding=None, stride=2):
        super(Fconv_Up, self).__init__()
        self.conv_up = fn.Fconv_Up(sizeP, inNum, outNum, tranNum, inP, output_padding, bias=True, stride=stride)
        self.bn_up = fn.F_BN(outNum, tranNum)
        self.relu_up = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu_up(x)
        return x


class Fconv_out(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, padding=None):
        super(Fconv_out, self).__init__()
        if padding is None:
            padding = sizeP // 2
        self.conv_out = fn.Fconv_PCA_out(sizeP, inNum, outNum, tranNum, inP, padding, ifIni=0, bias=True)
        self.bn_out = nn.BatchNorm2d(outNum)
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_out(x)
        x = self.bn_out(x)
        x = self.relu_out(x)
        return x


class InFConv(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, padding=None):
        super(InFConv, self).__init__()
        self.conv1 = Fconv(sizeP, inNum, outNum, tranNum, inP, padding, ifIni=1)
        self.conv2 = Fconv(sizeP, outNum, outNum, tranNum, inP, padding, ifIni=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FDown(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, padding=None, stride=2):
        super(FDown, self).__init__()
        self.conv1 = Fconv_Down(sizeP, inNum, outNum, tranNum, inP, padding, stride)
        self.conv2 = Fconv(sizeP, outNum, outNum, tranNum, inP, padding, ifIni=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class FUp(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, output_padding=None, stride=2):
        super(FUp, self).__init__()
        self.tranNum = tranNum
        self.up = Fconv_Up(sizeP, inNum // 2, inNum // 2, tranNum, inP, output_padding, stride)
        self.conv3 = Fconv(sizeP, inNum, outNum, tranNum, inP, ifIni=0)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        B1, C1, H1, W1 = x1.shape
        B2, C2, H2, W2 = x2.shape
        x1 = x1.reshape([B1, C1 // self.tranNum, self.tranNum, H1, W1])
        x2 = x2.reshape([B2, C2 // self.tranNum, self.tranNum, H2, W2])

        x = torch.cat([x2, x1], dim=1)
        x = x.reshape([x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]])

        x = self.conv3(x)
        return x


class finalFConv(nn.Module):
    def __init__(self, sizeP, inNum, outNum, tranNum=4, inP=None, padding=None):
        super(finalFConv, self).__init__()
        self.finalconv = fn.Fconv_PCA_out(sizeP, inNum, outNum, tranNum, inP, padding)
        self.bn = nn.BatchNorm2d(outNum)

    def forward(self, x):
        x = self.finalconv(x)
        x = self.bn(x)
        return x


class unet_fconv(nn.Module):
    def __init__(self, sizeP=5, in_channels=3, classes=2, tranNum=4, inP=None):
        super(unet_fconv, self).__init__()

        self.inc = InFConv(7, in_channels, 32, tranNum, inP)
        self.down1 = FDown(sizeP, 32, 64, tranNum, inP, 2, 2)
        self.down2 = FDown(sizeP, 64, 128, tranNum, inP, 2, 2)
        self.down3 = FDown(sizeP, 128, 256, tranNum, inP, 2, 2)
        self.down4 = FDown(sizeP, 256, 256, tranNum, inP, 2, 2)
        self.up1 = FUp(sizeP, 512, 128, tranNum, inP, None, 2)
        self.up2 = FUp(sizeP, 256, 64, tranNum, inP, None, 2)
        self.up3 = FUp(sizeP, 128, 32, tranNum, inP, None, 2)
        self.up4 = FUp(sizeP, 64, 32, tranNum, inP, None, 2)
        self.outc = finalFConv(sizeP, 32, classes, tranNum, inP, 2)

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
    net = unet_fconv()
    for name, param in net.named_parameters():
        nn.init.normal_(param, -0.01, 0.02)
    output = net(img)
    # print(output)
    img1 = torch.rot90(img, dims=(-1, -2))
    output1 = net(img1)
    output1 = torch.rot90(output1, k=-1, dims=(-1, -2))
    difference = output1 - output
    print(difference)