import os
import random


    # try:
    #
    #     train_f = open("train_1.txt", "x")
    #     eval_f = open("val_1", "x")
    #     train_f.write(train_files)
    #     eval_f.write(val_files)
    #     train_f.close()
    #     eval_f.close()
    # except FileExistsError as e:
    #     print(e)
    #     exit(1)

def main():
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    # files_path = "C:/Users/wei43/Downloads/VOC2012/Annotations"
    files_path = r"C:\Users\wei43\OneDrive\data_set\annotations"
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    test_rate = 0.1

    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    test_index = random.sample(range(0, files_num), k=int(files_num*test_rate))
    trainval_files = []
    test_files = []
    for index, file_name in enumerate(files_name):
        if index in test_index:
            test_files.append(file_name)
        else:
            trainval_files.append(file_name)

    try:
        # train_f = open("train.txt", "x")
        # eval_f = open("val.txt", "x")
        train_f = open("trainval.txt", "w")
        eval_f = open("test.txt", "w")
        train_f.write("\n".join(trainval_files))
        eval_f.write("\n".join(test_files))
        train_f.close()
        eval_f.close()
    except FileExistsError as e:
        print(e)
        exit(1)

    with open('trainval.txt', 'r') as f:
        lines = f.readlines()
        lens = len(lines)
        # print(lines, '\n', len)
        rate = 0.2
        val_index = random.sample(range(0, lens), k=int(lens * rate))
        train_files = []
        val_files = []
        with open('train_1.txt', 'w') as train_f:
            with open('val_1.txt', 'w') as val_f:
                for index, line in enumerate(lines):
                    if index in val_index:
                        val_f.write(line)
                    else:
                        train_f.write(line)

if __name__ == '__main__':
    main()
