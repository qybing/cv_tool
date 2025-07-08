import os
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import time
import imghdr

# ======== 配置区域 (按需修改这些变量) ========
SOURCE_DIR = 'JPEGImages'  # 要扫描的源目录路径
DUPLICATE_DIR = "images_duplicate"  # 备份目录路径（重复文件将移到这里）
USE_FAST_HASH = True  # 是否使用快速哈希模式 (True/False)
MIN_FILE_SIZE = 4096  # 最小处理的图片大小(字节)

import os
import hashlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import imghdr
import time


# ======== 配置区域 ========
# SOURCE_DIR = "/path/to/your/image/folder"  # 源图片文件夹路径
# DUPLICATE_DIR = "/path/to/duplicate/folder"  # 重复文件保存路径
# USE_FAST_HASH = True  # 使用快速哈希模式 (True/False)
# MIN_FILE_SIZE = 4096  # 最小处理的图片大小(字节)


# =========================

def is_valid_image(file_path):
    """验证文件是否为有效图片"""
    try:
        # 验证文件扩展名和实际格式匹配
        if imghdr.what(file_path) is None:
            return False
        # 快速验证图片文件头
        with open(file_path, 'rb') as f:
            header = f.read(16)
            return header.startswith(b'\xFF\xD8') or header.startswith(b'\x89PNG') or \
                header.startswith(b'GIF') or header.startswith(b'\x49\x49') or \
                header.startswith(b'MM')
    except (OSError, IOError):
        return False


def get_file_signature(file_path):
    """获取文件的哈希签名（快速模式或完整模式）"""
    # 创建自定义哈希对象（BLAKE2比MD5/SHA更快）
    hasher = hashlib.blake2b(digest_size=16)

    if USE_FAST_HASH:
        # 快速模式 - 采样读取关键部分
        size = os.path.getsize(file_path)
        with open(file_path, 'rb') as f:
            # 文件开头
            hasher.update(f.read(4096))

            # 文件中段（如果文件足够大）
            if size > 8192:
                f.seek(size // 2 - 2048)
                hasher.update(f.read(4096))

            # 文件末尾（如果文件足够大）
            if size > 12288:
                f.seek(-4096, 2)
                hasher.update(f.read(4096))
    else:
        # 完整模式 - 逐块读取整个文件
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(128 * 1024), b''):
                hasher.update(chunk)

    return hasher.hexdigest()


def find_image_duplicates():
    """扫描目录并查找重复图片"""
    print(f"📂 开始扫描图片目录: {SOURCE_DIR}")
    start_time = time.time()

    # 第一步：按大小预分组（快速筛选）
    size_map = defaultdict(list)
    file_count = 0

    for root, _, files in os.walk(SOURCE_DIR):
        for name in files:
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue

            try:
                size = os.path.getsize(path)
                if size < MIN_FILE_SIZE:
                    continue

                if not is_valid_image(path):
                    continue

                size_map[size].append(path)
                file_count += 1
            except OSError:
                continue

    print(f"🔍 发现 {file_count} 个有效图片文件（大小 > {MIN_FILE_SIZE} 字节）")
    print(f"⏱ 初始扫描耗时: {time.time() - start_time:.2f} 秒")

    # 第二步：并行计算哈希签名
    print(f"🧮 开始计算文件签名（{'快速模式' if USE_FAST_HASH else '完整模式'}）...")
    hash_map = defaultdict(list)
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        futures = {}
        # 只处理可能有重复的大小组
        for size, paths in size_map.items():
            if len(paths) > 1:
                for path in paths:
                    futures[executor.submit(get_file_signature, path)] = path

        # 收集结果
        for i, future in enumerate(futures):
            try:
                path = futures[future]
                file_hash = future.result()
                hash_map[file_hash].append(path)

                if (i + 1) % 500 == 0:
                    print(f"🔄 处理进度: {i + 1}/{len(futures)} 个文件")
            except Exception as e:
                print(f"⚠️ 处理失败 [{path}]: {str(e)}")

    # 第三步：识别重复项
    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}

    print(f"✅ 去重分析完成！发现 {len(duplicates)} 组重复图片")
    print(f"⏱ 总耗时: {time.time() - start_time:.2f} 秒")
    return duplicates


def move_duplicate_files(duplicates):
    """移动重复文件到指定目录"""
    if not duplicates:
        print("🎉 未发现重复图片")
        return 0, 0

    print(f"🚚 开始移动重复文件到: {DUPLICATE_DIR}")
    os.makedirs(DUPLICATE_DIR, exist_ok=True)

    moved_count = 0
    space_freed = 0

    for file_hash, paths in duplicates.items():
        # 按路径排序（保留第一个）
        paths.sort()
        keep_file = paths[0]
        move_files = paths[1:]

        for src_path in move_files:
            try:
                # 在目标位置保持相同目录结构
                rel_path = os.path.relpath(src_path, SOURCE_DIR)
                dest_path = os.path.join(DUPLICATE_DIR, rel_path)
                dest_dir = os.path.dirname(dest_path)

                # 创建目标目录
                os.makedirs(dest_dir, exist_ok=True)

                # 处理文件名冲突
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(dest_path)
                    counter = 1
                    while os.path.exists(f"{base}_{counter}{ext}"):
                        counter += 1
                    dest_path = f"{base}_{counter}{ext}"

                # 获取文件大小并移动
                file_size = os.path.getsize(src_path)
                shutil.move(src_path, dest_path)

                moved_count += 1
                space_freed += file_size

                # 进度报告
                if moved_count % 100 == 0:
                    print(f"📦 已移动 {moved_count} 个文件 | 释放空间: {space_freed / 1024 / 1024:.2f} MB")

            except Exception as e:
                print(f"❌ 移动失败 [{src_path}]: {str(e)}")

    return moved_count, space_freed


def main():
    print("\n" + "=" * 50)
    print(f"🖼️ 图片去重工具 | 源目录: {SOURCE_DIR}")
    print("=" * 50 + "\n")

    start_time = time.time()

    # 查找重复项
    duplicates = find_image_duplicates()

    # 移动重复文件
    move_start = time.time()
    moved_count, space_freed = move_duplicate_files(duplicates)
    move_time = time.time() - move_start

    # 结果汇总
    total_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("✨ 操作完成！结果汇总")
    print("=" * 50)
    print(f"🔸 发现重复组: {len(duplicates)} 组")
    print(f"🔸 移动文件数: {moved_count} 个")
    print(f"🔸 释放空间: {space_freed / 1024 / 1024:.2f} MB")
    print(f"🔸 移动耗时: {move_time:.2f} 秒")
    print(f"🔸 总耗时: {total_time:.2f} 秒")
    print(f"🔸 重复文件位置: {DUPLICATE_DIR}")

    if moved_count > 0:
        print("\n💡 提示: 请检查重复文件目录确认无误后，可安全删除")


def create_dir(file_path):
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    else:
        # 删除文件夹及其内容
        shutil.rmtree(file_path)
        # 重新创建空文件夹
        os.makedirs(file_path)


def file_move3():
    img_duplicate_dir = 'JPEGImages_duplicate'
    src_xml_dir = r'VOC/Annotations'
    xml_duplicate_dir = r'VOC/Annotations_duplicate'
    create_dir(xml_duplicate_dir)
    img_dict = {}
    img_duplicate_list = os.listdir(img_duplicate_dir)
    src_xml_dir_list = os.listdir(src_xml_dir)
    for img_name in img_duplicate_list:
        img_prefix = os.path.splitext(img_name)
        img_dict[img_prefix[0]] = img_name
    move_count = 0
    for xml_name in src_xml_dir_list:
        xml_prefix = os.path.splitext(xml_name)
        if xml_prefix[0] in img_dict:
            src_xml_path = os.path.join(src_xml_dir,xml_name)
            save_xml_path = os.path.join(xml_duplicate_dir,xml_name)
            shutil.move(src_xml_path,save_xml_path)
            move_count+=1
    print(f"重复的图片有:{len(img_duplicate_list)},移动的xml有:{move_count}")
    




if __name__ == "__main__":
    main()
