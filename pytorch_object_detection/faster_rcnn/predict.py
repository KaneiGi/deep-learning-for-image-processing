import os
import time
import json

import cv2
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_box

image_root = 'C:/Users/wei43/OneDrive/data_set/results/'
def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=91)

    # load train weights
    train_weights = "C:/Users/wei43/OneDrive/deep-learning-for-image-processing/pytorch_object_detection/faster_rcnn/backbone/fasterrcnn_resnet50_fpn_coco.pth"
    # train_weights = "C:/Users/wei43/Downloads/resNetFpn-model-14.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    # model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.load_state_dict(torch.load(train_weights, map_location=device))
    model.to(device)

    # # read class_indict
    # label_json_path = './pascal_voc_classes.json'
    label_json_path = '../train_coco_dataset/coco_91.json'
    # label_json_path = 'C:/Users/wei43/OneDrive/data_set/classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r',encoding='utf-8')
    class_dict = json.load(json_file)
    category_index = {int(k):v for k, v in class_dict.items()}
    # category_index = {v: k for k, v in class_dict.items()}
    # load image
    original_img = Image.open("./test.jpg")

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)
        cap = cv2.VideoCapture(0)
        while True:

            original_image = cap.read()[1]
            image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            image = data_transform(image)
            # image = data_transform(original_image)
            image = torch.unsqueeze(image, dim=0)
            # t_start = time_synchronized()
            predictions = model(image.to(device))[0]
            # t_end = time_synchronized()
            # print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            original_image = Image.fromarray(original_image)
            draw_box(original_image,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=0.5,
                     line_thickness=3)

            cv2_img = np.asarray(original_image)
            # original_image = np.array(original_image)
            cv2.imshow("capture", cv2_img)
            # cv2.imshow('image', img)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                # 通过esc键退出摄像
                cv2.destroyAllWindows()
                break
            elif key == ord("s"):
                # 通过s键保存图片，并退出。
                print(image_root + time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()) + '.png')
                cv2.imwrite(image_root + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.png', cv2_img)
                time.sleep(1)
            # plt.imshow(original_image)
            # plt.show()
            # 保存预测的图片结果
            # original_img.save("test_result.jpg")


if __name__ == '__main__':
    main()
    # train_weights = "C:/Users/wei43/OneDrive/deep-learning-for-image-processing/pytorch_object_detection/faster_rcnn/backbone/fasterrcnn_resnet50_fpn_coco.pth"
    # for key in torch.load(train_weights):
    #     print(key)
