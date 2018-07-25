import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

root = 'C:/tianchi/dataset/final-rank/Images/coat_length_labels'

img_paths = glob.glob('{}/*.{}'.format(root, 'jpg'))
infos_list = []
# 统计所有图片w, h信息
for img_path in img_paths:
    img = Image.open(img_path)
    w, h = img.size
    infos_list.append((w, h))

# 转为np数组，二维数组
infos = np.array(infos_list)

# 不重复w, h
uni_infos = np.array(list(set(infos_list)))
# print(uni_infos)

# 将(w, h)映射到idx
uni_infos2idx = {(uni_infos[i][0], uni_infos[i][1]): i for i in range(len(uni_infos))}
# print(uni_infos2idx)

a = [uni_infos2idx[(infos[i][0], infos[i][1])] for i in range(len(infos))]

a = np.array(a)
# 求和统计
uni_infos_sum = {k: (a == v).sum() for k, v in uni_infos2idx.items()}
print(uni_infos_sum)

# 找出字典中值最大对应的key
key = max(uni_infos_sum, key=lambda x: uni_infos_sum[x])
print(key, uni_infos_sum[key])

# 给字典value排序 reverse=True为降序
uni_infos_sum = sorted(uni_infos_sum.items(), key=lambda x: x[1], reverse=True)
print(uni_infos_sum)
