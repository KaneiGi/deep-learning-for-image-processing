import torch
import torchvision.transforms as transforms
from PIL import Image
from model_test import LeNet_test

mean = [0.49139965, 0.48215845, 0.44653094]
std = [0.24703224, 0.24348514, 0.26158786]
saved_path = "C:/Users/wei43/Downloads/CIFAR10/LeNet_test.pth"

transform = transforms.Compose(
	[transforms.Resize((32,32)),
	 transforms.ToTensor(),
	 transforms.Normalize(mean,std)]
)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

model = LeNet_test()
model.load_state_dict(torch.load(saved_path))

im = Image.open("1.jpg")
im = transform(im).unsqueeze(dim=0)

with torch.no_grad():
	output = model(im)
	predict = output.max(dim=-1)[1].numpy()
	predict_percent = torch.softmax(output,dim=-1)
print(classes[int(predict)])
print(predict_percent)


# a = torch.tensor([2,3,4])
# print(a.data.numpy())

