import torch.nn as nn
import torch
import torch.nn.functional as F


# class BasicConv(nn.Module):
#     def __init__(self,input_channels,output_channels,):
#     super().__init__()


# def test(a=10,b=20):
#     c =a+b
#     print(c,type(c))
#
# def wrapper(*args,**kwargs):
#     for kwarg in kwargs:
#         print(kwarg,type(kwarg))
#     print(type({**kwargs}))
#     # a,b,c = **kwargs
#     return test(*args,**kwargs)
#
# wrapper(a=10,b=20)

class BasicConv(nn.Module):
    def __init__(self, input_channels, output_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, **kwargs)

    def forward(self, x):
        out = F.relu(self.conv(x))
        return out


class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # self.average_pool = nn.AvgPool2d(5, 3)
        self.conv = BasicConv(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = F.avg_pool2d(x, 5, 3)
        # out = self.average_pool(out)
        out = self.conv(out)
        out = torch.flatten(out, start_dim=1)
        out = F.dropout(out, 0.5, training=self.training)
        out = self.fc1(out)
        out = F.relu(out, True)
        out = F.dropout(out, 0.5, training=self.training)
        out = self.fc2(out)
        return out


class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3reduce, ch3x3, ch5x5reduce, ch5x5, pool_proj):
        super().__init__()
        self.branch1 = BasicConv(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv(in_channels, ch3x3reduce, kernel_size=1),
            BasicConv(ch3x3reduce, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            BasicConv(in_channels, ch5x5reduce, kernel_size=1),
            BasicConv(ch5x5reduce, ch5x5, kernel_size=5, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        output = [branch1, branch2, branch3, branch4]
        return torch.cat(output, 1)


class GoogleNetTest(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogleNetTest, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv(3, 64, kernel_size=7, stride=2, padding=3)

        self.conv2 = BasicConv(64, 64, kernel_size=1)
        self.conv3 = BasicConv(64, 192, kernel_size=3, padding=1)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, kernel_size=3, stride=2, ceil_mode=True)

        out = self.conv2(out)
        out = self.conv3(out)
        out = F.max_pool2d(out, kernel_size=3, stride=2, ceil_mode=True)

        out = self.inception3a(out)
        out = self.inception3b(out)
        out = F.max_pool2d(out, kernel_size=3, stride=2, ceil_mode=True)

        out = self.inception4a(out)
        if self.training and self.aux_logits:
            aux1 = self.aux1(out)

        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        if self.training and self.aux_logits:
            aux2 = self.aux2(out)

        out = self.inception4e(out)
        out = F.max_pool2d(out, kernel_size=3, stride=2, ceil_mode=True)

        out = self.inception5a(out)
        out = self.inception5b(out)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        out = F.dropout(out, 0.5, training=self.training)
        out = self.fc(out)

        if self.training and self.aux_logits:
            return out, aux1, aux2

        return out
