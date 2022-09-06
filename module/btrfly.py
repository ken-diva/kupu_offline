import torch
import torch.nn as nn

class blue(nn.Module):
    def __init__(self, channels):
        super(blue, self).__init__()
        self.fwd = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[0], out_channels=channels[1],
                kernel_size=3, padding='same',
            ),
            nn.BatchNorm2d(num_features=channels[1]),
            nn.ReLU()
        )

    def forward(self, inp):
        out = self.fwd(inp)
        return out


class yellow(nn.Module):
    def __init__(self, channels, drop_out=False):
        super(yellow, self).__init__()
        self.fwd = nn.Sequential(
            # nn.Conv2d(
            #     in_channels=channels[0], out_channels=channels[0],
            #     kernel_size=1, padding='same',
            # ),
            nn.Conv2d(
                in_channels=channels[0], out_channels=channels[1],
                kernel_size=3, padding='same',
            ),
            # nn.Conv2d(
            #     in_channels=channels[1], out_channels=channels[1],
            #     kernel_size=1, padding='same',
            # ),
            nn.BatchNorm2d(num_features=channels[1]),
            nn.ReLU(),
            nn.Dropout(0.5 if drop_out else 0)
        )

    def forward(self, inp):
        out = self.fwd(inp)
        return out


class purple(nn.Module):
    def __init__(self, channels):
        super(purple, self).__init__()
        self.fwd = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=channels, out_channels=channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),
        )

    def forward(self, inp_main, inp_side):
        out = self.fwd(inp_main)
        out = torch.cat([inp_side, out], dim=1)
        return out


class red(nn.Module):
    def __init__(self):
        super(red, self).__init__()
        self.fwd = nn.Sequential(
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, inp):
        out = self.fwd(inp)
        return out


class orange(nn.Module):
    def __init__(self, channels):
        super(orange, self).__init__()
        self.fwd = nn.Sequential(
            nn.Conv2d(
                in_channels=channels[0], out_channels=channels[1],
                kernel_size=1, padding='same',
            ),
        )

    def forward(self, inp):
        out = self.fwd(inp)
        return out


class inp_arm(nn.Module):
    def __init__(self):
        super(inp_arm, self).__init__()
        self.blue = blue([3,  32])
        self.yellow0 = yellow([32,  32])
        self.yellow1 = yellow([32,  64])
        self.yellow2 = yellow([64, 128])
        self.red = red()

    def forward(self, inp):
        # print(inp.shape)
        out0 = self.blue(inp)
        # print(out0.shape)
        out1 = self.yellow0(out0)
        # print(out1.shape)
        out2 = self.red(out1)
        # print(out2.shape)

        out2 = self.yellow1(out2)
        # print(out2.shape)
        out3 = self.red(out2)
        # print(out3.shape)

        out3 = self.yellow2(out3)
        # print(out3.shape)
        outm = self.red(out3)
        # print(outm.shape)

        return outm, out1, out2, out3


class out_arm(nn.Module):
    def __init__(self, n_class=13):
        super(out_arm, self).__init__()
        self.purple0 = purple(512)
        self.purple1 = purple(256)
        self.purple2 = purple(128)
        self.yellow0 = yellow([128+512,      256])
        self.yellow1 = yellow([64+256,      128])
        self.yellow2 = yellow([32+128,       64])
        self.orangef = orange([64,  n_class])

    def forward(self, inpm, inp1, inp2, inp3):

        out = self.purple0(inpm, inp3)
        # print(out.shape)
        out = self.yellow0(out)
        # print(out.shape)

        out = self.purple1(out, inp2)
        # print(out.shape)
        out = self.yellow1(out)
        # print(out.shape)

        out = self.purple2(out, inp1)
        # print(out.shape)
        out = self.yellow2(out)
        # print(out.shape)
        out = self.orangef(out)
        # print(out.shape)

        return out


class body(nn.Module):
    def __init__(self):
        super(body, self).__init__()
        self.purple0 = purple(1024)
        self.purple1 = purple(512)
        self.yellow0 = yellow([256,  256], True)
        self.yellow1 = yellow([256,  512], True)
        self.yellow2 = yellow([512, 1024])
        self.yellow3 = yellow([512+1024,  512])
        self.yellow4 = yellow([256+512,  512])
        self.red = red()

    def forward(self, ant, pos):

        inp = torch.cat([ant, pos], dim=1)
        out1 = self.yellow0(inp)
        # print(out1.shape)

        out2 = self.red(out1)
        # print(out2.shape)
        out2 = self.yellow1(out2)
        # print(out2.shape)

        out = self.red(out2)
        # print(out.shape)
        out = self.yellow2(out)
        # print(out.shape)

        out = self.purple0(out, out2)
        # print(out.shape)
        out = self.yellow3(out)
        # print(out.shape)

        out = self.purple1(out, out1)
        # print(out.shape)
        out = self.yellow4(out)
        # print(out.shape)

        return out


class BtrflyNet(nn.Module):
    def __init__(self):
        super(BtrflyNet, self).__init__()
        self.inp_arm_ant = inp_arm()
        self.inp_arm_pos = inp_arm()
        self.out_arm_ant = out_arm()
        self.out_arm_pos = out_arm()
        self.body = body()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, ant, pos):

        # print('ant. arm')
        ant_body, ant1, ant2, ant3 = self.inp_arm_ant(ant)
        # print('pos. arm')
        pos_body, pos1, pos2, pos3 = self.inp_arm_pos(pos)

        # print('body')
        body_out = self.body(ant_body, pos_body)

        # print('ant. arm out')
        out_ant = self.out_arm_ant(body_out, ant1, ant2, ant3)
        # print('pos. arm out')
        out_pos = self.out_arm_pos(body_out, pos1, pos2, pos3)

        return self.softmax(out_ant), self.softmax(out_pos)
