#! python3
# _*_ coding: utf-8 _*_
# @Time : 2023/7/19 15:33 
# @Author : Jovan
# @File : voc_filter_class.py
# @desc :

import json
import os
import time
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import matplotlib.pyplot as plt


def gen_xml_2(all_class, json_data, root_dir, img1):
    xml_name = os.path.splitext(os.path.basename(img1))[0] + '.xml'
    img_name = img1
    # img_name = os.path.splitext(os.path.basename(img1))[0] + '.jpg'
    json_names = os.path.splitext(os.path.basename(img1))[0] + '.json'
    doc = Document()
    annotation = doc.createElement("annotation")
    doc.appendChild(annotation)

    folder = doc.createElement("folder")
    annotation.appendChild(folder)
    folder.appendChild(doc.createTextNode('train_imgs'))

    filename = doc.createElement("filename")
    annotation.appendChild(filename)
    filename.appendChild(doc.createTextNode(img_name))

    path = doc.createElement("path")
    annotation.appendChild(path)
    path.appendChild(doc.createTextNode(img_name))

    source = doc.createElement("source")
    annotation.appendChild(source)
    database = doc.createElement("database")
    source.appendChild(database)
    database.appendChild(doc.createTextNode('Unknown'))

    size = doc.createElement("size")
    annotation.appendChild(size)

    width = doc.createElement("width")
    size.appendChild(width)
    try:
        width.appendChild(doc.createTextNode(str(json_data['size']['width'])))
    except Exception as e:
        print(f"width error:{json_names}")
        width.appendChild(doc.createTextNode(str(1920)))

    height = doc.createElement("height")
    size.appendChild(height)
    try:
        height.appendChild(doc.createTextNode(str(json_data['size']['height'])))
    except Exception as e:
        height.appendChild(doc.createTextNode(str(1080)))
        print(f"height error:{json_names}")

    depth = doc.createElement("depth")
    size.appendChild(depth)
    depth.appendChild(doc.createTextNode('3'))

    segmented = doc.createElement("segmented")
    annotation.appendChild(segmented)
    segmented.appendChild(doc.createTextNode('0'))
    try:
        objects = json_data.get('outputs', None)
        if objects is None:
            objects = []
        if objects and isinstance(objects, dict):
            objects = objects.get('object', None)
    except Exception as e:
        objects = []
        print(f" objects error:{json_names}")
    is_continue = False
    for i in objects:
        name_ = i['name'].strip()
        all_class.add(name_)
        obj = doc.createElement("object")
        annotation.appendChild(obj)

        name = doc.createElement("name")
        name.appendChild(doc.createTextNode(str(name_)))

        pose = doc.createElement("pose")
        pose.appendChild(doc.createTextNode('Unspecified'))

        truncated = doc.createElement("truncated")
        truncated.appendChild(doc.createTextNode('0'))

        occluded = doc.createElement("occluded")
        occluded.appendChild(doc.createTextNode('0'))

        difficult = doc.createElement("difficult")
        difficult.appendChild(doc.createTextNode('0'))

        bndbox = doc.createElement("bndbox")

        obj.appendChild(name)
        obj.appendChild(pose)
        obj.appendChild(truncated)
        obj.appendChild(occluded)
        obj.appendChild(difficult)
        obj.appendChild(bndbox)
        obj.appendChild(bndbox)

        xmin = doc.createElement("xmin")
        xmin.appendChild(doc.createTextNode(str(round(i['bndbox']['xmin']))))

        ymin = doc.createElement("ymin")
        ymin.appendChild(doc.createTextNode(str(round(i['bndbox']['ymin']))))

        xmax = doc.createElement("xmax")
        xmax.appendChild(doc.createTextNode(str(round(i['bndbox']['xmax']))))

        ymax = doc.createElement("ymax")
        ymax.appendChild(doc.createTextNode(str(round(i['bndbox']['ymax']))))

        bndbox.appendChild(xmin)
        bndbox.appendChild(ymin)
        bndbox.appendChild(xmax)
        bndbox.appendChild(ymax)
    if not is_continue:
        with open(os.path.join(root_dir, xml_name), 'w') as f:
            f.write(doc.toprettyxml(indent="  "))


def scan_pic(x, y,name):
    # 绘制散点图
    plt.scatter(x, y)

    # 添加标题和标签
    plt.title(f'{name}')
    plt.xlabel('W')
    plt.ylabel('H')

    # 显示图形
    plt.show()


def bar_pic(x, y):
    bar_width = 0.2

    # 绘制柱状图
    plt.bar(x, y, width=bar_width)

    # 添加标题和标签
    plt.title('class count')
    plt.xlabel('class')
    plt.ylabel('count')
    # 显示每个柱子上的数值
    for i in range(len(x)):
        plt.text(x[i], y[i], str(y[i]), ha='center', va='bottom')
    # 调整x轴刻度
    plt.xticks([i for i in range(len(x))], x)
    # 显示图形
    plt.show()


def analyze_xml(src_dir,classes):
    '''
    分析xml文件
    :param src_dir:
    :return:
    '''
    src_xml_list = os.listdir(src_dir)
    count = 0
    all_class = set()
    width_list = []
    height_list = []
    box_w_list = []
    box_h_list = []
    classes_count = {}
    for xmlfile in src_xml_list:
        count += 1
        path_ = os.path.join(src_dir, xmlfile)
        tree = ET.parse(path_)
        root = tree.getroot()
        img_path = root.find('filename').text
        size = root.find('size')
        height = int(float(size.find('height').text))
        width = int(float(size.find('width').text))
        width_list.append(width)
        height_list.append(height)
        for object in root.iter('object'):
            name = object.find('name').text.strip()
            if name not in classes_count:
                classes_count[name] = 1
            else:
                classes_count[name] = classes_count[name] + 1
            all_class.add(name)
            bndbox = object.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            box_w_list.append(xmax-xmin)
            box_h_list.append(ymax-ymin)
    print(f"width_list:{len(width_list)},height_list:{len(height_list)}")
    name_list = []
    count_list = []
    for key, value in classes_count.items():
        name_list.append(key)
        count_list.append(value)
    # scan_pic(width_list, height_list,'PIC_W_H')
    scan_pic(box_w_list, box_h_list,'BOX_W_H')
    bar_pic(name_list, count_list)


def xml_filter_class(src_dir, target_dir, classes):
    src_xml_list = os.listdir(src_dir)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    count = 0
    all_class = set()
    for xmlfile in src_xml_list:
        count += 1
        path_ = os.path.join(src_dir, xmlfile)
        tree = ET.parse(path_)
        root = tree.getroot()
        img_path = root.find('filename').text
        size = root.find('size')
        try:
            height = int(float(size.find('height').text))
        except:
            height = 1080
        try:
            width = int(float(size.find('width').text))
        except:
            width = 1920
        json_data = dict()
        list_object = []
        json_data['path'] = img_path
        json_data['size'] = {'width': width, 'height': height}
        json_data['outputs'] = {'object': list_object}
        for object in root.iter('object'):
            name = object.find('name').text.strip()
            if name not in classes:
                continue
            bndbox = object.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            list_object.append(dict(name=name, bndbox=dict(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)))
        json_data['outputs'] = {'object': list_object}
        gen_xml_2(all_class, json_data, target_dir, os.path.basename(img_path))
    print(all_class)


if __name__ == '__main__':
    src_dir = r'F:\desk\look\xml'
    target_dir = r'F:\desk\look\new_xml'
    classes = ['']
    # xml_filter_class(src_dir, target_dir, classes)
    analyze_xml(target_dir,classes)
