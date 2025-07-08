#! python3
# _*_ coding: utf-8 _*_
# @Time : 2025/7/4 18:34 
# @Author : Jovan
# @File : yolo2voc.py
# @desc :

import os
import cv2
import logging
from xml.etree import ElementTree as ET
from xml.dom import minidom
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int, int]]:
    """快速获取图片尺寸信息，不加载完整图片"""
    try:
        # 方法1: 使用OpenCV仅读取元数据
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is not None:
            h, w = img.shape[:2]
            channels = img.shape[2] if len(img.shape) > 2 else 1
            return w, h, channels

        # 方法2: 回退到PIL（需要安装Pillow）
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.width, img.height, len(img.getbands())
        except ImportError:
            pass

        logger.warning(f"无法获取图片尺寸: {image_path}")
        return None
    except Exception as e:
        logger.error(f"图片尺寸获取错误 {image_path}: {str(e)}")
        return None


def yolo_to_voc(x_center: float, y_center: float,
                width: float, height: float,
                img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """YOLO归一化坐标转VOC绝对坐标"""
    x_center *= img_w
    y_center *= img_h
    w = width * img_w
    h = height * img_h

    xmin = max(0, int(x_center - w / 2))
    ymin = max(0, int(y_center - h / 2))
    xmax = min(img_w - 1, int(x_center + w / 2))
    ymax = min(img_h - 1, int(y_center + h / 2))

    return xmin, ymin, xmax, ymax


def process_single_file(txt_path: str, img_dir: str, xml_dir: str,
                        class_dict: Dict[str, str], img_exts: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')):
    """处理单个YOLO标注文件"""
    base_name = os.path.splitext(os.path.basename(txt_path))[0]

    # 查找图片文件（支持多种格式）
    img_path = None
    for ext in img_exts:
        test_path = os.path.join(img_dir, f"{base_name}{ext}")
        if os.path.exists(test_path):
            img_path = test_path
            break

    if not img_path:
        logger.warning(f"图片未找到: {base_name}，尝试过的扩展名: {img_exts}")
        return False

    # 获取图片尺寸
    dimensions = get_image_dimensions(img_path)
    if not dimensions:
        return False
    img_w, img_h, depth = dimensions

    # 创建XML结构
    annotation = ET.Element('annotation')

    ET.SubElement(annotation, 'folder').text = 'VOC_dataset'
    ET.SubElement(annotation, 'filename').text = os.path.basename(img_path)

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img_w)
    ET.SubElement(size, 'height').text = str(img_h)
    ET.SubElement(size, 'depth').text = str(depth)

    # 解析YOLO标注
    try:
        with open(txt_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"文件读取失败 {txt_path}: {str(e)}")
        return False

    for line_num, line in enumerate(lines, 1):
        data = line.strip().split()
        if len(data) != 5:
            logger.debug(f"跳过无效行 {txt_path}:{line_num} - {line.strip()}")
            continue

        try:
            class_id = data[0]
            x_center, y_center, w, h = map(float, data[1:])
            xmin, ymin, xmax, ymax = yolo_to_voc(x_center, y_center, w, h, img_w, img_h)

            # 添加对象节点
            obj = ET.SubElement(annotation, 'object')
            ET.SubElement(obj, 'name').text = class_dict.get(class_id, 'unknown')
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'

            bndbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bndbox, 'xmin').text = str(xmin)
            ET.SubElement(bndbox, 'ymin').text = str(ymin)
            ET.SubElement(bndbox, 'xmax').text = str(xmax)
            ET.SubElement(bndbox, 'ymax').text = str(ymax)

        except Exception as e:
            logger.error(f"解析错误 {txt_path}:{line_num} - {str(e)}")

    # 美化输出并保存XML
    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
    xml_output = os.path.join(xml_dir, f"{base_name}.xml")

    try:
        with open(xml_output, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        return True
    except Exception as e:
        logger.error(f"XML写入失败 {xml_output}: {str(e)}")
        return False


def convert_yolo_to_voc(txt_dir: str, img_dir: str, xml_dir: str,
                        class_dict: Dict[str, str], max_workers: int = None):
    """
    高效并行转换YOLO格式到VOC格式

    :param txt_dir: YOLO标注目录
    :param img_dir: 图片目录
    :param xml_dir: 输出XML目录
    :param class_dict: 类别映射字典
    :param max_workers: 最大并发线程数
    """
    os.makedirs(xml_dir, exist_ok=True)
    txt_files = [os.path.join(txt_dir, f) for f in os.listdir(txt_dir) if f.endswith('.txt')]
    total_files = len(txt_files)

    if not total_files:
        logger.warning("未找到YOLO标注文件(.txt)")
        return

    logger.info(f"开始转换 {total_files} 个文件...")

    # 线程池处理
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for txt_path in txt_files:
            future = executor.submit(
                process_single_file,
                txt_path, img_dir, xml_dir, class_dict
            )
            futures.append(future)

        for i, future in enumerate(futures):
            if future.result():
                success_count += 1
            if (i + 1) % 100 == 0 or (i + 1) == total_files:
                logger.info(f"进度: {i + 1}/{total_files} | 成功: {success_count}")

    logger.info(f"转换完成! 成功: {success_count}/{total_files} | 失败: {total_files - success_count}")


if __name__ == "__main__":
    # 配置参数
    CLASS_MAPPING = {'0': 'smog'}
    TXT_DIR = r'labels'
    IMG_DIR = r'images'
    XML_DIR = r'xmls'

    # 启动转换（max_workers=None自动设置最佳线程数）
    convert_yolo_to_voc(
        txt_dir=TXT_DIR,
        img_dir=IMG_DIR,
        xml_dir=XML_DIR,
        class_dict=CLASS_MAPPING,
        max_workers=os.cpu_count() * 2  # 建议2-4倍CPU核心数
    )
