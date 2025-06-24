import os
import argparse
import json
from datetime import datetime
from typing import Dict, List
from lxml import etree

from tqdm import tqdm
import re
import chardet


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result['encoding']


def get_label2id(labels_path: str) -> Dict[str, int]:
    """id is 1 start"""
    with open(labels_path, 'r') as f:
        labels_str = [line.strip() for line in f]
        # labels_str = f.read().split()
    labels_ids = list(range(0, len(labels_str) + 1))
    return dict(zip(labels_str, labels_ids))


def get_annpaths(ann_dir_path: str = None,
                 ann_ids_path: str = None,
                 ext: str = '',
                 annpaths_list_path: str = None) -> List[str]:

    # If use annotation paths list
    if annpaths_list_path is not None:
        with open(annpaths_list_path, 'r', encoding='utf-8') as f:
            ann_paths = f.read().split()
        return ann_paths

    # If use annotaion ids list
    ext_with_dot = '.' + ext if ext != '' else ''
    ann_ids = []
    with open(ann_ids_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            ann_id = line.strip()
            ann_ids.append(ann_id)
    ann_paths = [os.path.join(ann_dir_path, aid + ext_with_dot) for aid in ann_ids]
    return ann_paths


def get_image_info(annotation_root, file_name, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is not None:
        filename = annotation_root.findtext('filename')
    elif path is None:
        filename = os.path.splitext(os.path.basename(file_name))[0]+'.jpg'
    else:
        filename = os.path.basename(path)
    filename = os.path.splitext(os.path.basename(file_name))[0] + os.path.splitext(filename)[-1]
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[0])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(float(bndbox.findtext('xmin'))) - 1
    ymin = int(float(bndbox.findtext('ymin'))) - 1
    xmax = int(float(bndbox.findtext('xmax')))
    ymax = int(float(bndbox.findtext('ymax')))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    # 获取当前时间并格式化显示
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d")
    output_json_dict = {
        "info": {
            "description": "COCO format dataset converted from VOC",
            "version": "1.0",
            "year": formatted_time[0:4],
            "contributor": "auto-conversion",
            "date_created": formatted_time
        },
        "licenses": [{"name": "Unknown"}],
        "images": [],
        # "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        try:
            encoding = detect_encoding(a_path)
            ann_tree = etree.parse(a_path, parser=etree.XMLParser(encoding=encoding))
            ann_root = ann_tree.getroot()
        except Exception as e:
            print(f"file:{a_path},error:{e}")
            continue
        # ann_tree = ET.parse(a_path)
        # ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root, file_name=a_path,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']

        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)
    if os.path.exists(output_jsonpath):
        os.remove(output_jsonpath)
    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def main():
    parser = argparse.ArgumentParser(
        description='This script support converting voc format xmls to coco format json')
    parser.add_argument('--ann_dir', type=str, default=r'/xx/VOC/Annotations',
                        help='path to annotation files directory. It is not need when use --ann_paths_list')
    parser.add_argument('--ann_ids', type=str,
                        default=r'/xx/VOC/ImageSets/Main/train.txt',
                        help='path to annotation files ids list. It is not need when use --ann_paths_list')

    parser.add_argument('--ann_paths_list', type=str, default=None,
                        help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')

    parser.add_argument('--labels', type=str, default=r'/xx/VOC/label.txt',
                        help='path to label list.')
    parser.add_argument('--output', type=str, default=r'/xxx/VOC/train.json',
                        help='path to output json file')
    parser.add_argument('--ext', type=str, default='xml', help='additional extension of annotation file')
    parser.add_argument('--extract_num_from_imgid', action="store_true",
                        help='Extract image number from the image filename')
    args = parser.parse_args()
    label2id = get_label2id(labels_path=args.labels)
    ann_paths = get_annpaths(
        ann_dir_path=args.ann_dir,
        ann_ids_path=args.ann_ids,
        ext=args.ext,
        annpaths_list_path=args.ann_paths_list
    )
    convert_xmls_to_cocojson(
        annotation_paths=ann_paths,
        label2id=label2id,
        output_jsonpath=args.output,
        extract_num_from_imgid=args.extract_num_from_imgid
    )

def xiufu_josn():
    import json
    json_path = r'/home/jovan/dataset/shanxi/detect/VOC/val.json'
    with open(json_path, 'r') as f:
        data = json.load(f)
    data['info'] = data.get('info', {"description": "Auto-generated info"})
    with open(json_path, 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    '''
    ann_dir        voc的annotations文件夹
    ann_ids        voc的ImageSets\Main\train.txt  训练的列表‘
    labels         标签列表
    output         输出的json的位置  
    '''
    main()
    # xiufu_josn()
