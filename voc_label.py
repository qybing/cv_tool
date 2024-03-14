import xml.etree.ElementTree as ET
import shutil
import os


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


def convert_annotation(image_id,ann_path,label_path):
    '''
    生成txt文件
    :param image_id:文件名称(不带后缀)
    :param ann_path: xml路径
    :param label_path: 生成txt的路径
    :return:
    '''
    ann_name = image_id + '.xml'
    txt_name = image_id + '.txt'
    in_file = open(f'{os.path.join(ann_path,ann_name)}')
    out_file = open(f'{os.path.join(label_path,txt_name)}', 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        try:
            difficult = obj.find('difficult').text
        except Exception as e:
            print(f"{e},name:{ann_name}")
            difficult = 0
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
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
        new_path = os.path.join(dst_img_path,src_img)
        shutil.copyfile(old_path,new_path)
    print(f'count:{len(src_img_list)},copy finish')

# yolo需要的文件地址
current_dir = os.getcwd()
label_path = r'labels'
dst_img_path = 'images'

src_img_path = 'img'
# 切分好的train.txt的位置
train_path = r'ImageSets/Main'
ann_path = r'xml'
sets = ['train', 'val', 'test']
# 类别
classes = ['']
print(f"classes:{len(classes)}")
if not os.path.exists(label_path):
    os.makedirs(label_path)
if not os.path.exists(dst_img_path):
    os.makedirs(dst_img_path)
copy_image(src_img_path,dst_img_path)

for image_set in sets:
    image_ids = open(os.path.join(train_path, image_set + '.txt')).read().strip().split()
    list_file = open('%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        name = convert_annotation(image_id,ann_path,label_path)
        list_file.write(f'{os.path.join(current_dir, dst_img_path,name)}\n')
    list_file.close()
