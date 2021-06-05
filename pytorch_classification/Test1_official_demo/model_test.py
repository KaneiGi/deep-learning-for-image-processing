import torch.nn as nn
import torch.nn.functional as F
import torch


class LeNet_test(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 16, 5)
		self.pool1 = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(16, 32, 5)
		self.pool2 = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(32 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		out = self.pool1(F.relu(self.conv1(x)))
		out = self.pool2(F.relu(self.conv2(out)))
		out = out.view(-1, 32 * 5 * 5)
		out = F.relu(self.fc1(out))
		out = F.relu(self.fc2(out))
		out = self.fc3(out)
		return out


if __name__ == "__main__":
	image = torch.rand([4,3, 32, 32])
	print(image.size())
	model = LeNet_test()
	print(model)
	output = model(image)
	print(output,output.size())
	print(torch.max(output,dim=-1))
	for step in range(10000):
		if step % 500 == 499:
			print(step)