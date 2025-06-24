import os
import glob
import random
from lxml import etree

import chardet

config = {
    "Annotation": "Annotations",
    "JPEGImages": "JPEGImages",
}

train_per = 0.9
valid_per = 0.1
test_per = 0.0

data_xml_list = glob.glob(os.path.join(config['Annotation'], '*.xml'))
random.seed(666)
random.shuffle(data_xml_list)
data_length = len(data_xml_list)

train_point = int(data_length * train_per)
train_valid_point = int(data_length * (train_per + valid_per))

train_list = data_xml_list[:train_point]
valid_list = data_xml_list[train_point:train_valid_point]
test_list = data_xml_list[train_valid_point:]

label = set()


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']


for xml_path in data_xml_list:
    try:
        encoding = detect_encoding(xml_path)
        tree1 = etree.parse(xml_path, parser=etree.XMLParser(encoding=encoding))
        tree = tree1.getroot()
    except Exception as e:
        print(xml_path)
        print(f"Error: {e}")

    for obj in tree.findall('object'):
        label.add(obj.find('name').text)
with open('ImageSets/Main/train.txt', 'w') as ftrain, \
        open('ImageSets/Main/val.txt', 'w') as fvalid, \
        open('ImageSets/Main/test.txt', 'w') as ftest, \
        open('label.txt', 'w') as flabel:
    for i in train_list:
        ftrain.write(os.path.splitext(os.path.basename(i))[0] + "\n")
    for j in valid_list:
        fvalid.write(os.path.splitext(os.path.basename(j))[0] + "\n")
    for k in test_list:
        ftest.write(os.path.splitext(os.path.basename(k))[0] + "\n")
    for l in label:
        flabel.write(l + "\n")

print(
    f"总数据量: {data_length}, 训练集: {len(train_list)}, 验证集: {len(valid_list)}, 测试集: {len(test_list)}, 标签: {len(label)}")
print(f"标签: {label},总共有{len(label)}个标签")
print("done!")
