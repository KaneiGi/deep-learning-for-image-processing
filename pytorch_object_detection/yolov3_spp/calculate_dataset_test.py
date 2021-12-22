import os

train_annotation_dir = 'C:/Users/wei43/OneDrive/yolo_data_set/train/labels'
val_annotation_dir = 'C:/Users/wei43/OneDrive/yolo_data_set/val/labels'
classes_label = 'C:/Users/wei43/OneDrive/yolo_data_set/my_data_labels.names'
cfg_path = './cfg/yolov3-spp.cfg'

assert os.path.exists(train_annotation_dir),'dir does not exist'
assert os.path.exists(val_annotation_dir),'dir does not exist'
assert os.path.exists(classes_label),'file does not exist'
assert os.path.exists(cfg_path),'file does not exist'

def calculate_data_txt(txt_path,dataset_dir):
    with open(txt_path,'w',encoding='utf-8') as w:
        for file_name in os.listdir(dataset_dir):
            if file_name == 'classes.txt':
                continue
            img_path = dataset_dir.replace('labels','images')+'/'+file_name.split('.')[0]+'.png'
            line = img_path + '\n'
            assert os.path.exists(img_path),'file : {} does not exist'.format(img_path)
            w.write(line)

def create_data_data(create_data_path,label_path,train_path,val_path,classes_info):

    with open(create_data_path,'w',encoding='utf-8') as w:
        w.write('classes = {}'.format(len(classes_info))+'\n')
        w.write('train = {}'.format(train_path)+'\n')
        w.write('valid = {}'.format(val_path)+'\n')
        w.write('names = {}'.format(label_path)+'\n')

def change_and_create_cfg_file(classes_info,save_cfg_path = 'C:/Users/wei43/OneDrive/yolo_data_set/cfg/my_yolov3.cfg'):

    filters_lines = [636,722,809]
    classes_lines = [643,729,816]
    with open(cfg_path,'r') as r:
        cfg_lines = r.readlines()

    for i in filters_lines:
        assert 'filters' in cfg_lines[i-1],'filters param is not in line:{}'.format(i-1)
        output_num = (5+len(classes_info))*3
        cfg_lines[i-1] = 'filters={}\n'.format(output_num)

    for i in classes_lines:
        assert 'classes' in cfg_lines[i-1],'classes param is not in line:{}'.format(i-1)
        cfg_lines[i-1] = "classes = {}\n".format(len(classes_info))

    with open(save_cfg_path,'w',encoding='utf-8') as w:
        w.writelines(cfg_lines)

def main():
    train_txt_path = r'C:/Users/wei43/OneDrive/yolo_data_set/data/my_train_data.txt'
    val_txt_path = r'C:/Users/wei43/OneDrive/yolo_data_set/data/my_val_data.txt'
    calculate_data_txt(train_txt_path,train_annotation_dir)
    calculate_data_txt(val_txt_path,val_annotation_dir)

    with open(classes_label,'r',encoding='utf-8') as r:
        classes_info = [line.strip() for line in r.readlines() if len(line.strip())>0]
    create_data_data(r'C:/Users/wei43/OneDrive/yolo_data_set/data/my_data.data',
                     classes_label,train_txt_path,val_txt_path,classes_info)
    change_and_create_cfg_file(classes_info)

if __name__ == '__main__':
    main()