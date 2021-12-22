import os
import json
import sys

import torch
from PIL import Image
from torch import optim
from torchvision import transforms, datasets
import matplotlib as plt
from tqdm import tqdm

from model_test import AlexNetTest
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}

# data_root_1 = os.path.abspath(os.path.join(os.getcwd(),"../.."))

image_path = "C:/Users/wei43/Downloads/deep_learning_data/flower_data"
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])
train_num = len(train_dataset)

flower_list = train_dataset.class_to_idx
# print([key + ":" + str(val) for key, val in flower_list.items()])
cal_dict = dict((val, key) for key, val in flower_list.items())

json_str = json.dumps(cal_dict, indent=4)
with open("class_indices.json", "w") as json_file:
    json_file.write(json_str)

batch_size = 32
number_of_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 0 if "win" in sys.platform else 8])
print("Using {} dataloader workers every process.".format(number_of_workers),"( 0 is for windows platform )")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True,num_workers = number_of_workers)

val_dataset = datasets.ImageFolder(root=os.path.join(image_path,"val"),transform=data_transform["val"])
val_num = len(val_dataset)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size= val_num,
                                             shuffle=False,num_workers=number_of_workers)

print("using {} images for training, {} images for validation. ".format(train_num,val_num))

model = AlexNetTest(5,True)
model_weight_path = "C:/Users/wei43/Downloads/deep_learning_data/AlexNetTest.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)

epochs =20
save_path = "C:/Users/wei43/Downloads/deep_learning_data/AlexNetTest.pth"
best_accuracy = 0.0
train_step = len(train_loader)

for epoch in range(1,epochs+1):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    for step,data in enumerate(train_bar):
        images,labels = data
        optimizer.zero_grad()
        outputs = model(images.to(device))
        loss = loss_function(outputs,labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch,epochs,loss)

    model.eval()
    accuracy = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_dataloader)
        for val_data in val_bar:
            val_images,val_labels = val_data
            outputs = model(val_images.to(device))
            _,predict = torch.max(outputs,dim=1)
            accuracy += torch.eq(predict,val_labels.to(device)).sum().item()

    val_accuracy_rate = accuracy/val_num
    print("[epoch %d] train_loss: %.3f val_accuracy: %.3f" %(epoch,running_loss/batch_size,val_accuracy_rate))

    if val_accuracy_rate > best_accuracy:
        best_accuracy = val_accuracy_rate
        torch.save(model.state_dict(),save_path)
print("Training Finished")




