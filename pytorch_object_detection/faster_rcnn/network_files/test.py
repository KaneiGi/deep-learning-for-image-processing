# import torch
import os
import shutil
import re

path = 'C:/Users/wei43/Downloads/coco_91.txt'
path1 = 'C:/Users/wei43/Downloads/coco_91.json'
with open(path,'r') as file:
    lines = file.readlines()
    # print(lines)
    for idx,line in enumerate(lines):
        new_line = re.match(r'^(\d+)\s([a-z-]+)\s([a-z-]+)',line)
        if new_line is None:
            print('None')
        print(new_line.group(2), new_line.group(3))
        if new_line.group(2) != new_line.group(3):
            with open(path1,'a+') as f:
                f.write('\"'+str(idx+1)+'\"'+':'+' '+'\"'+new_line.group(2)+' '+new_line.group(3)+'\"'+','+'\n')
        else:
            with open(path1,'a+') as f:
                f.write('\"'+str(idx+1)+'\"'+':'+' '+'\"'+new_line.group(2)+'\"'+','+'\n')
        # print(new_line)
        # new_line = new_line.group(0)
        # with open(path1,'a+') as f:
        #     f.write(new_line+'\n')
# path = 'C:/Users/wei43/Downloads/image_data/'
#
# # path1 = 'C:/Users/wei43/Downloads/image_data/part_of_image/'
# #
# # files = os.listdir(path)
# # os.mkdir(path1)
# # print(files)
# # for idx, file in enumerate(files):
# #     if idx % 10 == 0:
# #         shutil.copy(path+file,path1+file)
# # a = torch.zeros((3,3,3))
#
# # print( a.size() == torch.tensor(a.size()),a.size(),torch.tensor(a.size()))
#
# a = torch.tensor([[0.0000, 0.0000, 0.0000,   0.0116, 0.0402, 0.0639]],
#        device='cuda:0')
# print(a)

# a = torch.arange(20).view(5,2,2)
# b = torch.tensor([0,4])[:,None]
# c = torch.tensor([1])[None,:]
# b1 = torch.tensor([0,4])[:,None]
# c1 = torch.tensor([1])[None,:]
# # print(a)
# # print(a[b1,c1].shape)
# # print(a[b1,c1])
#
# # a = torch.tensor([1,2,3])
# # b = torch.tensor([0,2])*(torch.ones((3,3,1))).to(torch.int64)
# # print(b,b.shape)
# # print(a[b],a[b].shape)
#
# a = torch.arange(10).view(2,-1)
# b = torch.tensor([0,1])[:,None]
# c= torch.tensor([0,2,4,])*(torch.ones((2,1))).to(torch.int64)
# # assert a[b] == a[c]
#
# print(a[b,c],a[b,c].shape)
# print(a[b],a[b].shape)
# print(a[:,c],a[:,c].shape)
# print(c,c.shape)
# print(a[:,c[0]],a[:,c[0]].shape)
# print(a[b].shape)
# print(a[c].shape)
