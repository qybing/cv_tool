import shutil
import os

import chardet
from lxml import etree


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = 1.0 if x * dw >= 1 else x * dw
    w = 1.0 if w * dw >= 1 else w * dw
    y = 1.0 if y * dh >= 1 else y * dh
    h = 1.0 if h * dh >= 1 else h * dh
    return (x, y, w, h)


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']


def convert_annotation(image_id, ann_path, label_path,class_count):
    '''
    生成txt文件
    :param image_id:文件名称(不带后缀)
    :param ann_path: xml路径
    :param label_path: 生成txt的路径
    :return:
    '''
    ann_name = image_id + '.xml'
    txt_name = image_id + '.txt'
    # in_file = open(f'{os.path.join(ann_path,ann_name)}',encoding='utf-8')
    out_file = open(f'{os.path.join(label_path, txt_name)}', 'w')
    try:
        encoding = detect_encoding(os.path.join(ann_path, ann_name))
        tree = etree.parse(os.path.join(ann_path, ann_name), parser=etree.XMLParser(encoding=encoding))
        root = tree.getroot()
    except Exception as e:
        print(f"file:{os.path.join(ann_path, ann_name)},error:{e}")
    try:
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        for obj in root.iter('object'):
            try:
                difficult = obj.find('difficult').text
            except Exception as e:
                difficult = '0'
                # print(f"difficult: {e},{ann_name}")
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            if cls not in class_count:
                class_count[cls] = 1
            else:
                class_count[cls] += 1
    except Exception as e:
        print(f"file:{os.path.join(ann_path, ann_name)},name:{root.find('filename').text}, error:{e}")
    return root.find('filename').text


def copy_image(src_img_path, dst_img_path):
    '''
    文件拷贝
    :param src_img_path:
    :param dst_img_path:
    :return:
    '''
    src_img_list = os.listdir(src_img_path)
    for src_img in src_img_list:
        old_path = os.path.join(src_img_path, src_img)
        new_path = os.path.join(dst_img_path, src_img)
        shutil.copyfile(old_path, new_path)
    print(f'count:{len(src_img_list)},copy finish')


def get_images_id(txt_path):
    images_id = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            images_id.append(line.strip())
    return images_id


# yolo需要的文件地址
current_dir = os.getcwd()
# current_dir = current_dir.replace('bag0228','bag0226')
label_path = r'labels'
dst_img_path = 'images'

src_img_path = 'JPEGImages'
# 切分好的train.txt的位置
train_path = r'ImageSets/Main'
ann_path = r'Annotations'
sets = ['train', 'val', 'test']
img_dict = {}
img_list = os.listdir(src_img_path)
for img_name in img_list:
    img_name_list = os.path.splitext(img_name)
    img_id = img_name_list[0]
    img_dict[img_id] = img_name

classes = ['smoke']
print(f"classes:{len(classes)}")
if not os.path.exists(label_path):
    os.makedirs(label_path)
if not os.path.exists(dst_img_path):
    os.makedirs(dst_img_path)
copy_image(src_img_path, dst_img_path)
class_count ={}
for image_set in sets:
    # image_ids = open(os.path.join(train_path, image_set + '.txt')).read().strip().split()
    image_ids = get_images_id(os.path.join(train_path, image_set + '.txt'))
    list_file = open('%s.txt' % (image_set), 'w')
    if image_set not in classes:
        class_count[image_set] = {}
    for image_id in image_ids:
        #print(f"image_id:{image_id}")
        name = convert_annotation(image_id, ann_path, label_path,class_count[image_set])
        #print(f"name:{name}")
        if name == None:
            name = image_id + '.jpg'
        name = image_id + os.path.splitext(name)[1]
        if os.path.splitext(name)[-1]=='.xml':
            name = img_dict[image_id]
        #print(f"name1:{name}")
        list_file.write(f'{os.path.join(current_dir, dst_img_path, name)}\n')
    list_file.close()
print(f"class_count:{class_count}")
