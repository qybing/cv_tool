#! python3
# _*_ coding: utf-8 _*_
# @Time : 2023/7/18 11:16 
# @Author : Jovan
# @File : split_data_1.py
# @desc :
# 导入的模块
import os
import glob
import random
import xml.etree.ElementTree as ET
config = {
    # Annotations path(Annotations 的文件夹路径)
    "Annotation":"Annotations",
    # JPEGImages path(JPEGImages 的文件夹路径)
    "JPEGImages":"img",
}
# 划分数据集

# 数据划分比例
# (训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1

# 按照比例划分数据集
train_per = 0.8
valid_per = 0.1
test_per = 0.1

data_xml_list = glob.glob(os.path.join(config['Annotation'], '*.xml'))
random.seed(666)
random.shuffle(data_xml_list)
data_length = len(data_xml_list)

train_point = int(data_length * train_per)
train_valid_point = int(data_length * (train_per + valid_per))

# 生成训练集，验证集, 测试集(8 : 1 : 1)
train_list = data_xml_list[:train_point]
valid_list = data_xml_list[train_point:train_valid_point]
test_list = data_xml_list[train_valid_point:]

# 生成label标签:
label = set()
for xml_path in data_xml_list:
        label = label | set([i.find('name').text for i in ET.parse(xml_path).findall('object')])


# 写入文件中
ftrain = open('ImageSets/Main/train.txt', 'w')
fvalid = open('ImageSets/Main/val.txt', 'w')
ftest = open('ImageSets/Main/test.txt', 'w')
flabel = open('label.txt', 'w')
for i in train_list:
        ftrain.write(os.path.splitext(os.path.basename(i))[0] + "\n")
for j in valid_list:
        fvalid.write(os.path.splitext(os.path.basename(j))[0] + "\n")
for k in test_list:
        ftest.write(os.path.splitext(os.path.basename(k))[0] + "\n")
for l in label:
        flabel.write(os.path.splitext(os.path.basename(l))[0] + "\n")
ftrain.close()
fvalid.close()
ftest.close()
flabel.close()
print("总数据量:{}, 训练集:{}, 验证集:{}, 测试集:{}, 标签:{}".format(len(data_xml_list), len(train_list), len(valid_list), len(test_list), len(label)))
print("done!")
