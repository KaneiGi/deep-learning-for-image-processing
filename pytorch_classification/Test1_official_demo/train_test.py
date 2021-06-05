import torch
import torchvision
import torch.nn as nn
from model_test import LeNet_test
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

data_path = "C:/Users/wei43/Downloads/CIFAR10"
# tensor_cifar10 = torchvision.datasets.CIFAR10(data_path, train=True, download=False,
#                                               transform=transforms.ToTensor())
# imgs = torch.stack([img_t for img_t, _ in tensor_cifar10], dim=-1)
# mean = imgs.view(3, -1).mean(dim=-1).numpy()
# std = imgs.view(3, -1).std(dim=-1).numpy()
# print(mean,std)
mean = [0.49139965, 0.48215845, 0.44653094]
std = [0.24703224, 0.24348514, 0.26158786]
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)])

# print(transform)
train_cifar10 = torchvision.datasets.CIFAR10(root=data_path, download=False,
                                             train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_cifar10, batch_size=32,
                                           shuffle=True, num_workers=0)

# img, label = transformed_cifar10[99]
# plt.imshow(img.permute(1, 2, 0))
# plt.show()

val_cifar10 = torchvision.datasets.CIFAR10(data_path, train=False,
                                           transform=transform)
val_loader = torch.utils.data.DataLoader(val_cifar10, batch_size=10000,
                                         shuffle=True, num_workers=0)
# for imgs,labels in train_loader:
#     print(imgs.size())
val_data_iter = iter(val_loader)
val_image, val_label = val_data_iter.next()

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# print(val_image.size(),val_label.size(),type(val_loader))
# print("".join("%10s" % classes[val_label[j]] for j in range(4)))
# for val_image in val_image:
#     val_image.squeeze(0)
#     plt.imshow(val_image.permute(1,2,0)[:,:,2])
#     plt.show()
#     print(val_image.permute(1,2,0)[:,:,2].size())

# print([(val_img,val_label)for val_img,val_label in val_loader])
# for ele,num in enumerate(val_loader):
#     print(ele)

model = LeNet_test()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

for epoch in range(10):
	running_loss = 0.0
	for step, data in enumerate(train_loader, start=0):
		inputs, labels = data
		optimizer.zero_grad()
		outputs = model(inputs)
		loss = loss_function(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if step % 500 == 499:  # 因为是从零开始
			with torch.no_grad():
				outputs = model(val_image)
				predict_y = torch.max(outputs, dim=-1)[1]
				accuracy = (predict_y == val_label).sum().item() / val_label.size(0)

				print("[%d,%5d] train_loss:%.3f test_accuracy:%.3f" %
				      (epoch + 1, step + 1, running_loss / 500, accuracy))
print("Training Finished")
save_path = "C:/Users/wei43/Downloads/CIFAR10/LeNet.pth"
torch.save(model.state_dict(),save_path)

