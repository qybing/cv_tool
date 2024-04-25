#! python3
# _*_ coding: utf-8 _*_
# @Author : Jovan
# @desc :
# Ultralytics YOLO 🚀, AGPL-3.0 license
import os
import time
from xml.dom.minidom import Document
from ultralytics import YOLO


def gen_xml_2(json_name, json_data, root_dir):
    xml_name = os.path.splitext(json_name)[0] + '.xml'
    img_name = json_name
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
    for i in objects:
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


def res_handle(result, Annotation, ig_class):
    # root_dir = r'/root/jovan/dataset/bag0229/Annotations'
    root_dir = Annotation
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    obj_list = []
    json_data = {}
    json_data['path'] = os.path.basename(result.path)
    json_data['time_labeled'] = int(time.time())
    json_data['labeled'] = True
    json_data['size'] = {'width': result.orig_img.shape[1], 'height': result.orig_img.shape[0],
                         'depth': result.orig_img.shape[-1]}
    xyxy_np = result.boxes.xyxy.numpy().astype(int)
    conf_np = result.boxes.conf.numpy().astype(float).round(3)  # confidence score, (N, 1)
    cls_np = result.boxes.cls.numpy().astype(int)
    for xyxy, conf, cls in zip(xyxy_np, conf_np, cls_np):
        object_ = {}
        object_['name'] = result.names[int(cls)]
        if len(ig_class) > 0 and object_['name'] in ig_class:
            continue
        object_['bndbox'] = {'xmin': xyxy[0], 'ymin': xyxy[1], 'xmax': xyxy[2], 'ymax': xyxy[3]}
        obj_list.append(object_)
        # print(xyxy)
        # print(conf)
        # print(cls)
    json_data['outputs'] = {'object': obj_list}
    gen_xml_2(os.path.basename(result.path), json_data, root_dir)


def predict_demo1():
    model_file = ''
    model = YOLO(model_file)  # load a custom model
    img_dir = 'JPEGImages'
    Annotation = 'Annotations'
    project = 'img'
    name = 'predict'
    # Predict with the model
    ig_class = []
    results = model(img_dir, save=False, imgsz=1280, project=project, iou=0.45,
                    name=name, device=0, hide_conf=True, conf=0.4)  # predict on an image
    for result in results:
        result = result.cpu()
        res_handle(result, Annotation, ig_class)
        # boxes = result.boxes


if __name__ == '__main__':
    predict_demo1()
