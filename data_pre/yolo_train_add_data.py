#! python3
# _*_ coding: utf-8 _*_
# @Time : 2025/7/2 17:23 
# @Author : Jovan
# @File : yolo_train_add_data.py
# @desc :
import os


def file_append(src_dir, dst_dir, cls_name):
    src_txt = os.path.join(src_dir, f"{cls_name}.txt")
    dst_txt = os.path.join(dst_dir, f"{cls_name}.txt")
    with open(src_txt, 'r') as f:
        src_content = f.readlines()
    with open(dst_txt, 'r') as f:
        dst_content = f.readlines()
    print(
        f'原有文件有{len(dst_content)}行, 新增文件有{len(src_content)}行, 合并后文件有{len(dst_content) + len(src_content)}行')
    for content in src_content:
        with open(dst_txt, 'a') as f:
            f.write(content)
    with open(dst_txt, 'r') as f:
        dst_content = f.readlines()
    print(f'新增后文件有{len(dst_content)}行')


def main():
    copy_class = ['train', 'val']
    src_dir = r'Objects365_v1/ImageSets/Main'
    dst_dir = r'train0707/ImageSets/Main'
    for cls_name in copy_class:
        print(f"复制{cls_name}文件开始")
        file_append(src_dir, dst_dir, cls_name)
        print(f"复制{cls_name}文件完成")
        print('*' * 50)


if __name__ == '__main__':
    # file_append()
    main()
