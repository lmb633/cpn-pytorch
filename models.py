import torch
import torch.nn as nn
from resnet import resnet50, resnet101, resnet152


class GlobalNet(nn.Module):
    def __init__(self, channel_sets, out_shape, num_class):
        super(GlobalNet, self).__init__()
        self.channel_sets = channel_sets
        laterals, upsamples, predicts = [], [], []
        self.layers = len(channel_sets)
        for i in range(self.layers):
            laterals.append(self._lateral(channel_sets[i]))
            predicts.append(self._predict(out_shape, num_class))
            if i != self.layers - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsample = nn.ModuleList(upsamples)
        self.predicts = nn.ModuleList(predicts)

    def _lateral(self, in_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def _upsample(self):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256)
        )

    def _predict(self, out_shape, num_class):
        return nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_class, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Upsample(size=out_shape, mode='bilinear', align_corners=True),
            nn.BatchNorm2d(num_class)
        )

    def forward(self, x):
        features, predicts = [], []
        for i in range(self.layers):
            feature = self.laterals[i](x[i])
            if i > 0:
                feature = feature + up
            features.append(feature)
            if i < self.layers - 1:
                up = self.upsample[i](feature)
            predicts.append(self.predicts[i](feature))
        return features, predicts


class Bottleneck(nn.Module):
    def __init__(self, in_channel, planes, stride=1, expansion=2):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channel, planes * 2, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * expansion)
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RefineNet(nn.Module):
    def __init__(self, lateral_channel=256, out_shape=(64, 48), num_class=15, cascade_num=4):
        super(RefineNet, self).__init__()
        self.cascade_num = cascade_num
        cascades = []
        for i in range(cascade_num):
            cascades.append(self._maker_layer(lateral_channel, cascade_num - i - 1, out_shape))
        self.cascades = nn.ModuleList(cascades)
        self.final_predict = self._predict(lateral_channel * cascade_num, num_class)

    def _maker_layer(self, in_channel, num, output_shape):
        layers = nn.Sequential()
        for i in range(num):
            layers.add_module('bottle', Bottleneck(in_channel, 128, expansion=2))
        layers.add_module('up', nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        return layers

    def _predict(self, in_channel, num_class):
        return nn.Sequential(
            Bottleneck(in_channel, 128, expansion=2),
            nn.Conv2d(256, num_class, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_class)
        )

    def forward(self, x):
        refine_feature = []
        for i in range(self.cascade_num):
            refine_feature.append((self.cascades[i](x[i])))
        out = torch.cat(refine_feature, dim=1)
        out = self.final_predict(out)
        return out


class CPN(nn.Module):
    def __init__(self, channel_sets=[2048, 1024, 512, 256], resnet='101', out_shape=(64, 48), n_class=15, pretrained=True):
        super(CPN, self).__init__()
        if resnet == '50':
            self.resnet = resnet50(pretrained)
        elif resnet == '101':
            self.resnet = resnet101(pretrained)
        else:
            self.resnet = resnet152(pretrained)
        self.global_net = GlobalNet(channel_sets=channel_sets, out_shape=out_shape, num_class=n_class)
        self.refine_net = RefineNet(lateral_channel=channel_sets[-1], out_shape=out_shape, num_class=n_class)

    def forward(self, x):
        feature = self.resnet(x)
        global_f, global_pred = self.global_net(feature)
        refine_pred = self.refine_net(global_f)
        return global_pred, refine_pred


if __name__ == '__main__':
    x = torch.zeros((1, 3, 128, 128))
    model = CPN(resnet='50', out_shape=(64, 64))
    out = model(x)
    print(out[0][0].shape)
