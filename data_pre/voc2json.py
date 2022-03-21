import json
import os
import time
import xml.etree.ElementTree as ET


def xml_to_json_third(root_dir):
    jpg_path = os.path.join(root_dir, 'JPEGImages')
    xml_path = os.path.join(root_dir, 'Annotations')
    json_path = os.path.join(root_dir, 'json')
    if not os.path.exists(json_path):
        os.mkdir(json_path)
    files = os.listdir(xml_path)
    count = 0
    for xmlfile in files:
        count += 1
        path_ = os.path.join(xml_path, xmlfile)
        tree = ET.parse(path_)
        root = tree.getroot()
        json_name = xmlfile.split('.')[0]
        img_path = os.path.join(jpg_path, json_name+".jpg")
        size = root.find('size')
        try:
            depth = int(float(size.find('depth').text))
        except:
            depth = 3
        height = int(float(size.find('height').text))
        width = int(float(size.find('width').text))
        list_object = []
        for object in root.iter('object'):
            name = object.find('name').text
            bndbox = object.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            list_object.append(dict(name=name, bndbox=dict(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)))

        json_dic = dict(path=img_path, outputs=dict(object=list_object), time_labeled=int(time.time()),
                        labeled=True, size=dict(width=width, height=height, depth=depth))

        print("已处理:{}".format(count))
        json_file_path = os.path.join(json_path, json_name) + ".json"
        with open(json_file_path, 'w') as f:
            json.dump(json_dic, f)


if __name__ == '__main__':
    # 指定VOC地址
    root_dir = r'G:\tmp\.so\VOC2007'
    xml_to_json_third(root_dir)
