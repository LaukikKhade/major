import torch
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.d1 = DoubleConv(3, 64)
        self.p1 = nn.MaxPool2d(2)

        self.d2 = DoubleConv(64, 128)
        self.p2 = nn.MaxPool2d(2)

        self.d3 = DoubleConv(128, 256)
        self.p3 = nn.MaxPool2d(2)

        self.d4 = DoubleConv(256, 512)
        self.p4 = nn.MaxPool2d(2)

        self.b = DoubleConv(512, 1024)

        self.u1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.c1 = DoubleConv(1024, 512)

        self.u2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.c2 = DoubleConv(512, 256)

        self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c3 = DoubleConv(256, 128)

        self.u4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c4 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        d4 = self.d4(self.p3(d3))

        b = self.b(self.p4(d4))

        u1 = self.u1(b)
        u1 = torch.cat([u1, d4], dim=1)
        u1 = self.c1(u1)

        u2 = self.u2(u1)
        u2 = torch.cat([u2, d3], dim=1)
        u2 = self.c2(u2)

        u3 = self.u3(u2)
        u3 = torch.cat([u3, d2], dim=1)
        u3 = self.c3(u3)

        u4 = self.u4(u3)
        u4 = torch.cat([u4, d1], dim=1)
        u4 = self.c4(u4)

        return self.out(u4)