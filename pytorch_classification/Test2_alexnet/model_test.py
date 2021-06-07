import torch.nn as nn
import torch


class AlexNetTest(nn.Module):
	def __init__(self, num_classes=1000, init_weights=True):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(3, 48, 11, 4, 2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(3, 2),
			nn.Conv2d(48, 128, 5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(3, 2),
			nn.Conv2d(128, 192, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(192, 192, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(192, 128, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(3, 2),
		)
		self.classifier = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(128 * 6 * 6, 2048),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5),
			nn.Linear(2048, 2048),
			nn.ReLU(inplace=True),
			nn.Linear(2048, num_classes)
		)
		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		out = self.features(x)
		out = torch.flatten(out, 1)
		out = self.classifier(out)
		return out

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
	model = AlexNetTest()
	# model_list = list(model.modules())
	# print(model_list,"\n",len(model_list))
	for module, num in enumerate(model.modules()):
		print(module, num)
