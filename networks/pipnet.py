import torch.nn as nn
import torch.nn.functional as F


class PIPNetResNet(nn.Module):
    def __init__(self, resnet, expansion, net_stride, num_lms, num_nb):
        super(PIPNetResNet, self).__init__()
        assert expansion in (1, 4)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.sigmoid = nn.Sigmoid()
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # calculate inplane & plane
        self.inplane = 512 * expansion  # 512/2048
        self.net_stride = net_stride
        self.num_lms = num_lms
        self.num_nb = num_nb
        self.plane = self.inplane // (net_stride // 32)  # 32
        # setup extra layers
        self._make_extra_layers(inplane=self.inplane, plane=self.plane)
        # setup det headers
        self._make_det_headers(plane=self.plane)

    def _make_extra_layers(
            self,
            inplane: int = 2048,
            plane: int = 2048
    ):
        assert self.net_stride in (32, 64, 128)
        if self.net_stride == 128:
            self.layer5 = nn.Conv2d(inplane, plane, kernel_size=(3, 3),
                                    stride=(2, 2), padding=(1, 1))
            self.bn5 = nn.BatchNorm2d(plane)
            self.layer6 = nn.Conv2d(plane, plane, kernel_size=(3, 3),
                                    stride=(2, 2), padding=(1, 1))
            self.bn6 = nn.BatchNorm2d(plane)
            # init
            nn.init.normal_(self.layer5.weight, std=0.001)
            if self.layer5.bias is not None:
                nn.init.constant_(self.layer5.bias, 0)
            nn.init.constant_(self.bn5.weight, 1)
            nn.init.constant_(self.bn5.bias, 0)

            nn.init.normal_(self.layer6.weight, std=0.001)
            if self.layer6.bias is not None:
                nn.init.constant_(self.layer6.bias, 0)
            nn.init.constant_(self.bn6.weight, 1)
            nn.init.constant_(self.bn6.bias, 0)
        elif self.net_stride == 64:
            self.layer5 = nn.Conv2d(
                inplane, plane, kernel_size=(3, 3),
                stride=(2, 2), padding=(1, 1)
            )
            self.bn5 = nn.BatchNorm2d(plane)
            # init
            nn.init.normal_(self.layer5.weight, std=0.001)
            if self.layer5.bias is not None:
                nn.init.constant_(self.layer5.bias, 0)
            nn.init.constant_(self.bn5.weight, 1)
            nn.init.constant_(self.bn5.bias, 0)

    def _make_det_headers(
            self,
            plane: int = 2048
    ):
        # cls_layer: (68,8,8)
        self.cls_layer = nn.Conv2d(
            plane, self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # x_layer: (68,8,8)
        self.x_layer = nn.Conv2d(
            plane, self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # y_layer: (68,8,8)
        self.y_layer = nn.Conv2d(
            plane, self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # nb_x_layer: (68*10,8,8)
        self.nb_x_layer = nn.Conv2d(
            plane, self.num_nb * self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # nb_y_layer: (68*10,8,8)
        self.nb_y_layer = nn.Conv2d(
            plane, self.num_nb * self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        nn.init.normal_(self.cls_layer.weight, std=0.001)
        if self.cls_layer.bias is not None:
            nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.nb_x_layer.weight, std=0.001)
        if self.nb_x_layer.bias is not None:
            nn.init.constant_(self.nb_x_layer.bias, 0)

        nn.init.normal_(self.nb_y_layer.weight, std=0.001)
        if self.nb_y_layer.bias is not None:
            nn.init.constant_(self.nb_y_layer.bias, 0)

    def _forward_extra(self, x):
        if self.net_stride == 128:
            x = F.relu(self.bn5(self.layer5(x)))
            x = F.relu(self.bn6(self.layer6(x)))
        elif self.net_stride == 64:
            x = F.relu(self.bn5(self.layer5(x)))

        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        self._forward_extra(x)
        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5
