
'''
# 指定VOC地址
# json格式转voc
'''

import json
import os
from xml.dom.minidom import Document, parse


def gen_xml_2(json_name, json_data, root_dir):
    xml_name = os.path.splitext(json_name)[0] + '.xml'
    img_name = os.path.splitext(json_name)[0] + '.jpg'
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
    path.appendChild(doc.createTextNode(os.path.join(root_dir, img_name)))

    source = doc.createElement("source")
    annotation.appendChild(source)
    database = doc.createElement("database")
    source.appendChild(database)
    database.appendChild(doc.createTextNode('Unknown'))

    size = doc.createElement("size")
    annotation.appendChild(size)

    width = doc.createElement("width")
    size.appendChild(width)
    width.appendChild(doc.createTextNode(str(json_data['size']['width'])))

    height = doc.createElement("height")
    size.appendChild(height)
    height.appendChild(doc.createTextNode(str(json_data['size']['height'])))

    depth = doc.createElement("depth")
    size.appendChild(depth)
    depth.appendChild(doc.createTextNode('3'))

    segmented = doc.createElement("segmented")
    annotation.appendChild(segmented)
    segmented.appendChild(doc.createTextNode('0'))
    objects = json_data['outputs']['object']
    ig_class = ['mask', 'helmet', 'hline', 'head', 'face']
    for i in objects:
        if i['name'] in ig_class:
            continue
        obj = doc.createElement("object")
        annotation.appendChild(obj)

        name = doc.createElement("name")
        name.appendChild(doc.createTextNode(str(i['name'])))

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
    with open(os.path.join(root_dir, xml_name), 'w') as f:
        f.write(doc.toprettyxml(indent="  "))


def convert():
    json_dir = r'G:\tmp\20220325\cut_data\data4\json'
    target_dir = r'G:\tmp\20220325\cut_data\data4\ann'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    json_list = os.listdir(json_dir)
    i = 0
    for json_name in json_list:
        with open(os.path.join(json_dir, json_name), 'r', encoding='utf8')as fp:
            json_data = json.load(fp)

        gen_xml_2(json_name, json_data, target_dir)
        i += 1
        if i % 100 == 0:
            print(f'已经完成{i}, 剩余{len(json_list) - i}')


if __name__ == '__main__':

    convert()
