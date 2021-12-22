import json
import os   

import matplotlib
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from model_test import AlexNetTest

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

img_path = "./tulip.jpg"
assert os.path.exists(img_path), "file: \"{}\" do not exist ".format(img_path)
img = Image.open(img_path)
plt.imshow(img)
plt.show()
img_transformed = data_transform(img)
img_transformed = torch.unsqueeze(img_transformed, dim=0)

json_path = "./class_indices.json"
assert os.path.exists(json_path), "file: \"{}\" do not exist ".format(json_path)
with open(json_path) as json_file:
    class_indices = json.load(json_file)
print(class_indices)
model = AlexNetTest(num_classes=5).to(device)

weights_path = "./AlexNetTest.pth"
assert os.path.exists(weights_path), "file: \"{}\" do not exist ".format(weights_path)
model.load_state_dict(torch.load(weights_path))

model.eval()
with torch.no_grad():
    output = model(img_transformed.to(device)).cpu()
    predict = torch.softmax(output, dim=-1)
    predict_index = torch.argmax(predict).item()

print("class : {} probability : {:.3}".format(class_indices[str(predict_index)],
                                              predict[0, predict_index].item()))
