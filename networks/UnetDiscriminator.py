import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = self.conv2(x)
        x += residual
        return x

class UNetDiscriminator(nn.Module):
    def __init__(self, in_ch, patch_size, base_ch=16, use_fp16=False):
        super(UNetDiscriminator, self).__init__()
        self.use_fp16 = use_fp16
        conv_dtype = torch.float16 if use_fp16 else torch.float32

        # layers = self.find_archi(patch_size)
        layers = [[3, 2], [3, 2], [3, 2], [3, 2], [3, 2]] #[3, 2]
        # print(layers)

        level_chs = {i-1: v for i, v in enumerate([min(base_ch * (2**i), 512) for i in range(len(layers) + 1)]) }

        self.in_conv = nn.Conv2d(in_ch, level_chs[-1], kernel_size=1, padding=0)

        self.convs = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for i, (kernel_size, stride) in enumerate(layers):
            self.convs.append(nn.Conv2d(level_chs[i-1], level_chs[i], kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2 ))
            
            self.upconvs.insert(0, nn.ConvTranspose2d( level_chs[i]*(2 if i != len(layers)-1 else 1), level_chs[i-1],
                                                        kernel_size=4, stride=stride, padding=1) )

        self.out_conv = nn.Conv2d(level_chs[-1] * 2, 1, kernel_size=1, padding=0)

        # self.center_out = nn.Conv2d(level_chs[len(layers) - 1], 16, kernel_size=1,stride = 2, padding=0)
        self.center_out = nn.Conv2d(level_chs[len(layers) - 1], 1, kernel_size=1,stride = 1, padding=0)
        self.center_conv = nn.Conv2d(level_chs[len(layers) - 1], level_chs[len(layers) - 1], kernel_size=1, padding=0)

    def forward(self, x):
        if self.use_fp16:
            x = x.to(torch.float16)

        x = F.leaky_relu(self.in_conv(x), negative_slope=0.2)

        encs = []
        for i, conv in enumerate(self.convs):
            encs.insert(0, x)
            x = F.leaky_relu(conv(x), negative_slope=0.2)
            # print(x.shape, "down")          

        center_out, x = self.center_out(x), F.leaky_relu(self.center_conv(x), negative_slope=0.2)
        # print(center_out.shape, x.shape, "center")

        for i, (upconv, enc) in enumerate(zip(self.upconvs, encs)):

            x = F.leaky_relu(upconv(x), negative_slope=0.2)
            x = torch.cat([enc, x], dim=1)
            # print(x.shape, "up")

        x = self.out_conv(x)

        if self.use_fp16:
            center_out = center_out.to(torch.float32)
            x = x.to(torch.float32)

        return center_out, x


    def calc_receptive_field_size(self, layers):
        rf = 0
        ts = 1
        for i, (k, s) in enumerate(layers):
            if i == 0:
                rf = k
            else:
                rf += (k-1)*ts
            ts *= s
        return rf

    def find_archi(self, target_patch_size, max_layers=9):

        s = {}
        for layers_count in range(1,max_layers+1):
            val = 1 << (layers_count-1)
            while True:
                val -= 1

                layers = []
                sum_st = 0
                layers.append ( [3, 2])
                sum_st += 2
                for i in range(layers_count-1):
                    st = 1 + (1 if val & (1 << i) !=0 else 0 )
                    layers.append ( [3, st ])
                    sum_st += st                

                rf = self.calc_receptive_field_size(layers)

                s_rf = s.get(rf, None)
                if s_rf is None:
                    s[rf] = (layers_count, sum_st, layers)
                else:
                    if layers_count < s_rf[0] or \
                    ( layers_count == s_rf[0] and sum_st > s_rf[1] ):
                        s[rf] = (layers_count, sum_st, layers)

                if val == 0:
                    break

        x = sorted(list(s.keys()))
        q=x[np.abs(np.array(x)-target_patch_size).argmin()]
        # print( q)
        return s[q][2]


if __name__=="__main__" :
    D = UNetDiscriminator(3, 16)
    x = torch.randn(12, 3, 512, 512)
    co, o =  D(x)
    # x = torch.zeros((1, 1, 64, 48))
    # for i in range(4):
    #     for j in range(3):
    #         x[0, 0, 16*i:16*(i + 1), 16 * j:16 * (j+1)] = co[i * 3 + j, 0, :, :] 
    
    print(x.shape)
    print(co.shape, o.shape)