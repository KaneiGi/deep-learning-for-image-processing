import torch.nn as nn
import torch


class VGGTest(nn.Module):
    def __init__(self, model_name="vgg16", num_classes=1000, init_weights=False):
        assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
        super().__init__()
        self.model_name = cfgs[model_name]
        self.features = self.make_features(self.model_name)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, start_dim=1)
        out = self.classifier(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight)
                nn.init.constant(m.bias, 0)


    def make_features(self,cfg: list):
        layers = []
        input_channels = 3
        for layer in cfg:
            if layer == "M":
                layers += [nn.MaxPool2d(2, 2)]
            else:
                conv2d = nn.Conv2d(input_channels, layer, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(True)]
                input_channels = layer

        return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

if __name__=="__main__":
    # for num,layer in enumerate(cfgs["vgg11"]):
    #     print(num,layer)
    model = VGGTest()
    for layer in model.modules():
        print(layer)