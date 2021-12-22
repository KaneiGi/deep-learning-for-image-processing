from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree


class VOC2012DataSetTest(Dataset):

    def __init__(self, voc_root, transform, txt_name: str = 'train.txt'):
        self.root = voc_root
        self.image_root = os.path.join(self.root, 'JPEGImages')
        self.annotation_root = os.path.join(self.root, 'Annotations')

        txt_path = os.path.join(self.root, 'ImageSets', 'Main', txt_name)
        assert os.path.exists(txt_path), '{} is not found'.format(txt_path)

        with open(txt_path) as read:
            self.xml_list = [os.path.join(self.annotation_root, line.strip() + '.xml')
                             for line in read.readlines()]
        assert len(self.xml_list) > 0, "in '{}' file does not find any information".format(txt_path)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), "'{}' is not found".format(xml_path)

        json_file = './pascal_voc_classes.json'
        assert os.path.exists(json_file), "'{}' does not found".format(json_file)
        with open(json_file) as json_path:
            self.class_dict = json.load(json_path)

        self.transforms = transform

    def __len__(self):
        return len(self.xml_list)

    def __getitem__(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)['annotation']
        img_path = os.path.join(self.image_root, data['filename'])
        image = Image.open(img_path)
        if image.format != 'JPEG':
            raise ValueError('Image {} format is not JPEG'.format(img_path))
        boxes = []
        labels = []
        iscrowd = []
        assert 'object' in data, '{} lack of object information'.format(xml_path)
        for obj in data['object']:
            xmin = float(obj['bndbox']['xmin'])
            xmax = float(obj['bndbox']['xmax'])
            ymin = float(obj['bndbox']['ymin'])
            ymax = float(obj['bndbox']['ymax'])

            if xmax <= xmin or ymax <= ymin:
                print('Warning: in {} xml, there are some bndbox w/h <=0'.format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj['name']])
            if 'difficult' in obj:
                iscrowd.append(int(obj['difficult']))
            else:
                iscrowd.append(0)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:3] - boxes[:1]) * (boxes[:2] - boxes[:0])

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['iscrowd'] = iscrowd
        target['area'] = area
        target['image_id'] = image_id

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        xml_path = self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = self.parse_xml_to_dict(xml)
        data_height = int(data['object']['height'])
        data_width = int(data['object']['width'])
        return data_height, data_width

    def parse_xml_to_dict(self, xml):
        if len(xml) == 0:
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:
                    result[child.tag] = []                          #dict -> list -> dict 
                result[child.tag].append(child_result[child.tag])
            return {xml.tag: result}

if __name__ == '__main__':
    a =[]
    for i in range(5):
        a.append([(i,i),i])
    a,b,c,d,e = a
    print(tuple(zip(a,b,c,d,e)))