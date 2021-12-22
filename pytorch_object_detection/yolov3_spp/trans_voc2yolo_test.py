import os
from tqdm import tqdm
from lxml import etree
import json
import shutil

data_root = r'C:\Users\wei43\OneDrive\data_set'

train_txt = 'train.txt'
val_txt = 'val.txt'

save_file_root = r'C:\Users\wei43\OneDrive\yolo_data_set'
label_json_path = r'C:\Users\wei43\OneDrive\data_set\classes.json'

images_path = os.path.join(data_root, 'images')
xmls_path = os.path.join(data_root, 'annotations')
train_txt_path = os.path.join(data_root, 'train.txt')
val_txt_path = os.path.join(data_root, 'val.txt')

assert os.path.exists(images_path), 'image path does not exist'
assert os.path.exists(xmls_path), 'xml path does not exist'
assert os.path.exists(train_txt_path), 'train txt path does not exist'
assert os.path.exists(val_txt_path), 'val txt path does not exist'

if os.path.exists(save_file_root) is False:
    os.mkdir(save_file_root)


def parse_xml_to_dict(xml):
    if len(xml) == 0:
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
            # result.update(child_result)
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def translate_info(file_names: list, save_root: str, class_dict: dict, train_val='train'):
    save_txt_path = os.path.join(save_root, train_val, "labels")
    if os.path.exists(save_txt_path) is False:
        os.makedirs(save_txt_path)

    save_image_path = os.path.join(save_root, train_val, "images")
    if os.path.exists(save_image_path) is False:
        os.makedirs(save_image_path)

    for file in tqdm(file_names, desc='translate {} file ...'.format(train_val)):

        image_path = os.path.join(images_path, file + '.png')
        assert os.path.exists(image_path), 'image does not exist'

        xml_path = os.path.join(xmls_path, file + '.xml')
        assert os.path.exists(xml_path), 'xml does not exist'

        with open(xml_path,'r+',encoding='utf-8') as xml_file:
            xml_str = xml_file.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)['annotation']
        img_height = int(data['size']['height'])
        img_width = int(data['size']['width'])

        with open(os.path.join(save_txt_path, file + '.txt'), 'w')as f:
            assert 'object' in data.keys(), "file: '{}' lack of object key.".format(xml_path)
            for index, obj in enumerate(data['object']):
                xmin = float(obj['bndbox']['xmin'])
                xmax = float(obj['bndbox']['xmax'])
                ymin = float(obj['bndbox']['ymin'])
                ymax = float(obj['bndbox']['ymax'])
                class_name = obj['name']
                class_index = class_dict[class_name] - 1

                if xmax <= xmin or ymax <= ymin:
                    print('Warning: in {} xml, some w or h <= 0'.format(xml_path))
                    continue

                xcenter = xmin + (xmax - xmin) / 2
                ycenter = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                xcenter = round(xcenter / img_width, 6)
                ycenter = round(ycenter / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)

                info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]

                if index == 0:
                    f.write(' '.join(info))
                else:
                    f.write('\n' + ' '.join(info))

        shutil.copyfile(image_path,
                        os.path.join(save_image_path, image_path.split(os.sep)[-1]))  # os.sep在win上面是\,在Linux上面是/


def create_class_names(class_dict: dict):
    keys = class_dict.keys()
    with open('C:/Users/wei43/OneDrive/yolo_data_set/my_data_labels.names', 'w',encoding='utf-8') as w:
        for index, k in enumerate(keys):
            if index + 1 == len(keys):
                w.write(k)
            else:
                w.write(k + '\n')


def main():
    with open(label_json_path, 'r', encoding='utf-8') as json_file:
        class_dict = json.load(json_file)

    with open(train_txt_path, 'r',encoding='utf-8') as train_file:
        train_file_names = [i for i in train_file.read().splitlines() if len(i.strip()) > 0]
    # f.readlines()和f.read().splitlines()都是返回一个list，f.readlines()后面有加\n, f.read().splitlines()没有\n

    translate_info(train_file_names, save_file_root, class_dict, 'train')

    with open(val_txt_path, 'r') as r:
        val_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]

    translate_info(val_file_names, save_file_root, class_dict, 'val')

    create_class_names(class_dict)


if __name__ == '__main__':
    main()
