import os
import hashlib
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import imghdr
import logging


class ImageDeduplicator:
    def __init__(self, src_dir, dup_dir, min_file_size=4096, fast_mode=True, keep_mode="first", max_workers=None):
        """
        初始化图片去重器

        :param src_dir: 源图片目录路径
        :param dup_dir: 重复文件存放目录路径
        :param min_file_size: 最小处理的文件大小（字节），默认为4KB
        :param fast_mode: 是否使用快速模式（采样哈希），默认为True
        :param keep_mode: 保留模式，可选 "first"（保留第一个）、"last"（保留最后一个）、"oldest"（最旧文件）
        :param max_workers: 最大并发线程数，None则使用CPU核心数*2
        """
        self.src_dir = os.path.abspath(src_dir)
        self.dup_dir = os.path.abspath(dup_dir)
        self.min_file_size = min_file_size
        self.fast_mode = fast_mode
        self.keep_mode = keep_mode
        self.max_workers = max_workers or (os.cpu_count() * 2)

        # 日志记录
        self.total_images = 0
        self.duplicate_groups = 0
        self.files_moved = 0
        self.space_freed = 0
        self.processing_time = 0

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("image_deduplication.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger()

        # 验证配置
        if not os.path.isdir(self.src_dir):
            raise ValueError(f"源目录不存在: {self.src_dir}")
        os.makedirs(self.dup_dir, exist_ok=True)

        # 支持的图片格式
        self.supported_formats = {"jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"}

    def is_valid_image(self, file_path):
        """验证文件是否为有效的图片格式并满足最小大小要求"""
        try:
            # 检查文件大小
            if os.path.getsize(file_path) < self.min_file_size:
                return False

            # 检查文件头
            with open(file_path, 'rb') as f:
                header = f.read(16)
                if not header:
                    return False

                # 使用imghdr检测格式
                file_format = imghdr.what(None, h=header)
                return file_format and file_format.lower() in self.supported_formats
        except (OSError, IOError):
            return False

    def calculate_file_hash(self, file_path):
        """计算文件的哈希值"""
        hasher = hashlib.blake2b(digest_size=16)
        file_size = os.path.getsize(file_path)

        # 对大文件使用采样策略
        if self.fast_mode and file_size > 1024 * 1024:  # >1MB的文件使用快速采样
            with open(file_path, 'rb') as f:
                # 文件头
                hasher.update(f.read(4096))

                # 文件中段
                if file_size > 8192:
                    f.seek(file_size // 2 - 2048)
                    hasher.update(f.read(4096))

                # 文件尾
                if file_size > 12288:
                    f.seek(-4096, 2)
                    hasher.update(f.read(4096))
        else:
            # 完整文件哈希
            with open(file_path, 'rb') as f:
                # 使用内存视图优化性能
                while chunk := f.read(65536):
                    hasher.update(chunk)

        return hasher.hexdigest()

    def process_image(self, file_info):
        """处理单个图像文件并计算哈希值"""
        file_path, file_size = file_info
        try:
            if self.is_valid_image(file_path):
                return self.calculate_file_hash(file_path), file_path, file_size
        except Exception as e:
            self.logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
        return None, None, None

    def find_duplicates(self):
        """查找目录中的重复图片"""
        start_time = time.time()
        self.logger.info(f"🔍 扫描源目录: {self.src_dir}")
        self.logger.info(f"⚡ 使用 {'快速' if self.fast_mode else '完整'} 模式检测重复")
        self.logger.info(f"📏 最小处理大小: {self.min_file_size} 字节")

        # 收集候选图片文件（文件路径 + 文件大小）
        candidate_images = []
        for root, _, files in os.walk(self.src_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                if not os.path.isfile(file_path):
                    continue

                try:
                    file_size = os.path.getsize(file_path)
                    # 首先检查文件大小是否达到最小要求
                    if file_size >= self.min_file_size:
                        candidate_images.append((file_path, file_size))
                except OSError as e:
                    self.logger.warning(f"无法获取文件大小: {file_path}: {str(e)}")

        self.total_images = len(candidate_images)
        if not candidate_images:
            self.logger.warning("⚠️ 没有找到符合条件的图片文件")
            return {}

        self.logger.info(f"📊 发现 {self.total_images} 个候选图片文件")

        # 并行处理所有候选图片
        hash_map = defaultdict(list)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_image, file_info) for file_info in candidate_images]

            for i, future in enumerate(futures):
                file_hash, file_path, file_size = future.result()
                if file_hash and file_path and file_size:
                    hash_map[file_hash].append((file_path, file_size))

                # 进度报告
                if i % 500 == 0 or i == len(candidate_images) - 1:
                    processed = i + 1
                    self.logger.info(f"🔄 处理中: {processed}/{self.total_images} ({processed / self.total_images:.1%})")

        # 识别重复项（哈希值相同的文件）
        duplicates = {}
        for file_hash, files in hash_map.items():
            if len(files) > 1:
                # 按路径排序确保可预测性
                duplicates[file_hash] = [file_path for file_path, _ in sorted(files)]
                self.duplicate_groups += 1

        self.processing_time = time.time() - start_time
        self.logger.info(f"✅ 发现 {self.duplicate_groups} 组重复图片")
        self.logger.info(f"⏱ 检测耗时: {self.processing_time:.2f}秒")

        return duplicates

    def move_duplicates(self, duplicates):
        """移动重复文件到指定目录"""
        if not duplicates:
            self.logger.info("🎉 未发现重复图片")
            return

        self.logger.info(f"🚚 开始移动重复文件到: {self.dup_dir}")
        start_time = time.time()
        moved_count = 0
        freed_space = 0

        # 为每个哈希组创建目标子目录
        for file_hash, files in duplicates.items():
            # 确定要保留的文件（保持原位置）
            if self.keep_mode == "first":
                keep_file = files[0]
                move_files = files[1:]
            elif self.keep_mode == "last":
                keep_file = files[-1]
                move_files = files[:-1]
            else:  # oldest 模式（最后修改时间最早）
                files_with_mtime = [(f, os.path.getmtime(f)) for f in files]
                files_with_mtime.sort(key=lambda x: x[1])
                keep_file = files_with_mtime[0][0]
                move_files = [f[0] for f in files_with_mtime[1:]]

            # 为当前哈希组创建目标子目录
            group_dir = os.path.join(self.dup_dir, f"group_{file_hash[:8]}")
            os.makedirs(group_dir, exist_ok=True)

            # 移动重复文件
            for file_path in move_files:
                try:
                    # 保持相对路径结构
                    rel_path = os.path.relpath(file_path, self.src_dir)
                    dest_path = os.path.join(group_dir, os.path.basename(rel_path))

                    # 处理文件名冲突
                    counter = 1
                    base, ext = os.path.splitext(dest_path)
                    while os.path.exists(dest_path):
                        dest_path = f"{base}_{counter}{ext}"
                        counter += 1

                    # 获取文件大小并移动
                    file_size = os.path.getsize(file_path)
                    shutil.move(file_path, dest_path)

                    freed_space += file_size
                    moved_count += 1
                    self.logger.debug(f"移动: {file_path} → {dest_path} ({file_size / 1024:.2f} KB)")

                    # 定期报告进度
                    if moved_count % 50 == 0:
                        self.logger.info(f"📦 已移动 {moved_count} 个文件 | "
                                         f"释放空间: {freed_space / 1024 / 1024:.2f} MB")
                except Exception as e:
                    self.logger.error(f"移动失败 {file_path}: {str(e)}")

        # 更新统计数据
        self.files_moved = moved_count
        self.space_freed = freed_space
        move_time = time.time() - start_time

        self.logger.info(f"📦 移动完成: {moved_count} 文件")
        self.logger.info(f"💾 释放空间: {freed_space:,} 字节 ({freed_space / 1024 / 1024:.2f} MB)")
        self.logger.info(f"⏱ 移动耗时: {move_time:.2f}秒")

    def generate_report(self):
        """生成处理结果报告"""
        report = [
            "\n" + "=" * 60,
            "📊 图片去重报告",
            "=" * 60,
            f"🔸 源目录: {self.src_dir}",
            f"🔸 重复文件目录: {self.dup_dir}",
            f"🔸 扫描图片总数: {self.total_images}",
            f"🔸 发现重复组: {self.duplicate_groups}",
            f"🔸 移动文件数: {self.files_moved}",
            f"🔸 释放空间: {self.space_freed:,} 字节 ({self.space_freed / 1024 / 1024:.2f} MB)",
            f"🔸 总处理时间: {self.processing_time:.2f} 秒",
            "",
            "💡 提示: 重复文件已按哈希组分组存放，请检查后删除",
            "=" * 60
        ]

        for line in report:
            self.logger.info(line)

        # 同时保存到文本文件
        with open("deduplication_report.txt", "w") as f:
            f.write("\n".join(line.replace("🔸 ", "").replace("💡 ", "").replace("📊 ", "").replace("=" * 60, "=" * 50)))


if __name__ == "__main__":
    # 配置参数
    SOURCE_DIR = "/home/jovan/dataset/shandong/smog/train0707/images_duplicate"  # 修改为您的源目录
    DUPLICATE_DIR = "/home/jovan/dataset/shandong/smog/train0707/duplicates"  # 修改为您的目标目录
    MIN_FILE_SIZE = 100  # 字节
    FAST_MODE = True
    KEEP_MODE = "first"  # "first", "last" or "oldest"

    # 执行去重
    try:
        deduper = ImageDeduplicator(
            src_dir=SOURCE_DIR,
            dup_dir=DUPLICATE_DIR,
            min_file_size=MIN_FILE_SIZE,
            fast_mode=FAST_MODE,
            keep_mode=KEEP_MODE
        )

        duplicates = deduper.find_duplicates()
        deduper.move_duplicates(duplicates)
        deduper.generate_report()
    except Exception as e:
        logging.error(f"❌ 程序错误: {str(e)}")
        raise
