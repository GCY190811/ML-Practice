import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBN(nn.Module):
    def __init__(self, in_c, out_c, k_size, k_stride, pad):
        super(ConvBN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=k_size, stride=k_stride, padding=pad)
        self.bn1 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return F.relu(self.bn1(self.conv1(x)))


class DepthWiseModule(nn.Module):
    def __init__(self, dw_in_c, pw_out_c, dw_stride=1):
        super(DepthWiseModule, self).__init__()

        self.dw_conv1 = nn.Conv2d(in_channels=dw_in_c, out_channels=dw_in_c, kernel_size=3, stride=dw_stride, padding=1, groups=dw_in_c)
        self.bn1 = nn.BatchNorm2d(dw_in_c)

        self.pw_conv2 = nn.Conv2d(in_channels=dw_in_c, out_channels=pw_out_c, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(pw_out_c)

    def forward(self, X):
        dw_out = F.relu(self.bn1(self.dw_conv1(X)))
        pw_out = F.relu(self.bn2(self.pw_conv2(dw_out)))
        return pw_out


def MobileNet():
    net = nn.Sequential(
        # stage 1
        # TODO: in_c change to 1 for fMnist
        ConvBN(1, 32, 3, 2, 1),

        # stage 2
        DepthWiseModule(32, 64),
        DepthWiseModule(64, 128, 2),
        DepthWiseModule(128, 128, 1),
        DepthWiseModule(128, 256, 2),
        DepthWiseModule(256, 256, 1),
        DepthWiseModule(256, 512, 2),

        # 5 times
        DepthWiseModule(512, 512, 1),
        DepthWiseModule(512, 512, 1),
        DepthWiseModule(512, 512, 1),
        DepthWiseModule(512, 512, 1),
        DepthWiseModule(512, 512, 1),

        DepthWiseModule(512, 1024, 2),
        DepthWiseModule(1024, 1024, 1),

        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),

        # TODO: change to 10 types for fMnist
        nn.Linear(1024, 10),
        # TODO: close Softmax() for using CrossEntropyLoss()
        # nn.Softmax(dim=1)
    )

    return net

if __name__ == "__main__":
    net = MobileNet()
    X = torch.rand((1, 3, 224, 224))

    print('fake input:', X.shape)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output size():\t', X.size())
