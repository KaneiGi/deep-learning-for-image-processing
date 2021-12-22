import os
import json
import pickle
import random

import matplotlib.pyplot as plt
import torch


def read_split_data_test(root: str, val_rate: float = 0.2, plot_image: bool = False):
    random.seed(0)
    assert os.path.exists(root), 'dataset root :{} dose not exit'.format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    flower_class.sort()

    with open('class_indices_test.json', 'w') as json_file:
        json.dump(dict((v, k) for v, k in enumerate(flower_class)), json_file, indent=4)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []

    every_class_num = []
    supported = ['.jpg', '.JPG', '.png', '.PNG']

    for cla in flower_class:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla_path, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = flower_class.index(cla)

        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)
    print('{} images were found in the dataset'.format(sum(every_class_num)))
    print('{} images for training'.format(len(train_images_path)))
    print('{} images for validation'.format(len(val_images_path)))

    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + 'does not exist'
    with open(json_path, 'r') as json_file:
        class_indices = json.load(json_file)
    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # if images.size()[0] < plot_num:
            #     break
            if list(labels[i:-1]) == []:
                continue
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


if __name__ == '__main__':
    # path = "C:/Users/wei43/Downloads/deep_learning_data/flower_data/train"
    # read_split_data_test(path, plot_image=True)
    # a = torch.zeros((3, 3, 3))
    # # print(list(a[5:-1]))
    # print(a.size()[0])
    json_path = './class_indices_test.json'
    with open(json_path) as json_file:
        class_dict = json.load(json_file)
        print(class_dict)
