#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""VOC 标注数据集 -> YOLO txt 清单式分层切分工具（train / val / test）。

综合两版旧脚本优点重写：
  承自 opencode/split_voc_to_yolo.py：
    - 命令行接口 + 完整参数校验（路径 / 类别 / 占比 / 阈值）
    - 稀有类优先轮次切分：类别按组数升序逐轮分配，保证稀有类在
      val/test 中有样本，类别不均衡时验证指标不失真
    - VOC 解析审计：difficult 可选过滤、未知类别默认报错（防静默漏标）、
      坏框 / 越界框分类统计
    - 生产级 pHash：LUT popcount 汉明距离、内存预算分块比对、Windows
      文件占用重试、解码失败图隔离为独立组
    - 输出目录覆盖保护（--overwrite）；详细结果报告 + 训练命令提示
  承自 omp_demo/split_dataset.py：
    - 防泄漏默认开启：连拍 / 视频抽帧的相邻帧用 pHash 聚成组，整组进
      同一子集，避免验证指标虚高（--no-dedup 可关）
    - XML 尺寸缺失 / 非法时用图像实际尺寸兜底，不丢标注
    - 框数桶分层：轮次池够大时按框数 0/1/2/3+ 分桶独立装填，保持各
      子集目标密度分布一致；大组配额容差，防止百张连拍组独占 val/test
    - txt 清单输出：不复制图像、零额外磁盘占用；labels 镜像在 images
      同级目录（Ultralytics 约定），总是重新生成保证与当前类别表一致
新增：
    - labels 覆盖保护：图像同级 labels 目录已有标注时必须 --overwrite
      才重写，防止换 --classes 重跑静默篡改旧标注
    - --force-split <SPLIT> <清单>：把指定图像（整组原子）强制分到指定
      子集，服务"误检/漏检批次主体进 train、抽部分进 val"的迭代流程
    - --max-group-size K：连通分量超过 K 张时按相似序拆分，防止长视频
      传递链把整段视频绑成巨组（巨组整组进 train 会让 val 对该场景失明）
    - --copy-out <DIR>：把自包含训练目录（images/、labels/ 子目录与
      data.yaml）拷贝到指定路径，可整体打包/换机直接训练；不传或为空
      则仅生成清单，不拷贝图像

用法示例（PowerShell）：
  # 不带任何参数：直接使用下方"默认数据集"配置（在可调参数集中区修改）
  python split_voc.py

  # 显式指定路径 / 类别 / 占比（覆盖默认值）
  python split_voc.py --images D:\data\images --anno D:\data\Annotations `
      --classes phone cigarette --ratios 0.8 0.1 0.1

  # 不要 test 集：test 占比给 0，或只给 2 个数（两种写法效果相同）
  python split_voc.py --ratios 0.9 0.1 0

  # 数据为独立照片（无连拍 / 抽帧）时关闭去重以加速：
  python split_voc.py ... --no-dedup

  # 误检/漏检新批次：强制整组进 train（清单每行一个文件名，支持 # 注释）
  python split_voc.py ... --force-split train hard_batch_stems.txt

  # 视频帧有固定文件名前缀时更省事，通配符直接圈定（无需清单文件）：
  python split_voc.py ... --force-split-glob train "batch0912_*"

  # 长视频抽帧：限制去重组最大 200 张，防止整段视频被绑成一个组
  python split_voc.py ... --max-group-size 200

  # 需要打包/换机训练时，把自包含训练目录（图像+标注+配置）拷贝出来：
  python split_voc.py ... --copy-out D:\data\yolo_ready

  # 典型误报批次迭代（30 张级相似帧，一条命令）：
  python split_voc.py --overwrite --max-group-size 5 --force-split-glob train "vid0912_*"

参数逐项说明：
  --images / --anno / --classes / --ratios / --output
      数据来源与去向，默认值在脚本顶部"可调参数集中区"，日常无需传。
      --ratios 给 2 个数 = 无 test；test 给 0 同效。

  --overwrite
      重跑开关。清单/data.yaml/labels 已存在时不带它会被拦截；
      改了标注或类别后重跑必带。

  --no-dedup
      关闭 pHash 相似分组。视频帧/连拍数据【不要关】——关了相邻帧会
      随机散进 train 和 val，验证指标虚高。独立照片可关以提速。

  --phash-threshold（默认 8，范围 0~63）
      "多像才算同组"的汉明距离阈值。报告发现不同场景被误并 → 调小
      (5~6)；同场景没并上 → 调大(10~12)。

  --max-group-size（默认 0 = 不限制）
      一个组最多几张。组 = pHash 判定的相似图像簇，是切分的原子单位：
      组内图像永远进同一子集，绝不拆开。
      【不传的后果】30 张相似帧会绑成一个 30 张的巨组，分配时整批
      all-or-nothing——实测可能整批进 val，模型一张都学不到。
      【建议值】批次张数 ÷ 10（30 张→3，40 张→4，几百帧长视频→100~200）；
      30~40 张批次直接用 5 也行。它只决定"切多细"，train/val 比例
      仍由 --ratios 决定；组越小比例越准，组越大防泄漏越强。

  --force-split-glob <SPLIT> <通配符>   （可重复传）
      文件名匹配通配符的图像【整组】强制进指定子集，免清单文件。
      如 --force-split-glob train "vid0912_*"

  --force-split <SPLIT> <清单文件>      （可重复传）
      同上，但用清单指定：每行一个文件名（带不带扩展名均可，支持
      # 注释）。两者可叠加，组合出"主体锁 train + 抽样锁 val"。
      同组图像被指向不同子集会直接报错（组不可拆）。

  --skip-difficult
      过滤 VOC 标注中 difficult=1 的目标。

  --unknown-classes error|skip（默认 error）
      XML 出现 --classes 未声明的类别时：error=报错退出（防漏标），
      skip=跳过该目标继续。

  --exclude-background
      无标注图像默认作为背景图（空标签）纳入切分，可降低误检；
      加此参数则排除。

  --copy-out <DIR>
      把自包含训练目录拷贝到 DIR（images/、labels/ 子目录 + data.yaml），
      整个文件夹可直接打包/换机训练。不传或传空 = 仅生成清单不拷贝。
      拷贝目标已有内容时需配合 --overwrite。

  --seed（默认 42）
      随机种子。同种子 + 同数据 = 切分完全一致，对比实验期间不要改。

输出（--output 默认 <图像文件夹>_split）：
  <output>/train.txt|val.txt|test.txt   清单，每行一张图的绝对路径
  <图像文件夹同级>/labels/*.txt          YOLO 标注（总是重新生成）
  <output>/data.yaml                     Ultralytics 训练配置

训练：
  yolo detect train model=yolo26n.pt data=<output>/data.yaml epochs=100 imgsz=640

依赖：pillow（图像尺寸兜底 / 去重）、numpy（仅去重）。
数据规整（XML 均含有效尺寸）且加 --no-dedup 时，纯标准库即可运行。
"""

import argparse
import fnmatch
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# 可调参数集中区：常规调整只改这里，无需翻代码
# ============================================================================

# ---- 默认数据集（不带参数运行时使用，命令行 --images/--anno/--classes 可覆盖）----
DEFAULT_IMAGES_DIR = Path(r'')
DEFAULT_ANNO_DIR = Path(r'')
DEFAULT_CLASSES = ['phone', 'cigarette']        # 顺序即 YOLO 类别 id（第 0 个是 id 0）
DEFAULT_COPY_OUT = r"E:\jovan\data\整理好的数据\标注好的数据\吸烟和玩手机数据集整合\train"                        # 自包含训练目录拷贝目标，如 r'D:\data\yolo_ready'；
                                               # None 或空串 = 仅生成清单，不拷贝图像

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}  # 参与的图像扩展名
SPLIT_NAMES = ('train', 'val', 'test')        # 子集名（Ultralytics 约定，勿改）

DEFAULT_RATIOS = (0.9, 0.1, 0)                 # 默认 train/val/test 占比（test 可为 0，为 0 时不生成 test.txt）
DEFAULT_SEED = 42                             # 默认随机种子（同种子结果可复现）
RATIO_SUM_TOLERANCE = 0.02                    # 占比之和偏离 1 的容忍度，超出仅提示并归一化

# ---- pHash 去重分组（默认开启，防连拍/抽帧数据泄漏）----
PHASH_IMG_SIZE = 32                           # 计算哈希前灰度图缩放尺寸
PHASH_LOW_FREQ = 8                            # 取 DCT 低频 8x8 系数 -> 64 位哈希
PHASH_MAX_DISTANCE = PHASH_LOW_FREQ ** 2 - 1  # 汉明距离合法上限（63）
PHASH_DEFAULT_THRESHOLD = 8                   # 默认"近似重复"阈值，视频抽帧建议 6~10
HASH_RETRY_TIMES = 3                          # 图像读取重试次数（应对杀软/索引器瞬时占用）
HASH_RETRY_SLEEP = 0.2                        # 重试等待基数（秒），第 n 次等 n * 该值

# ---- 相似度比对性能（图像上万张时才敏感）----
COMPARE_BUDGET_CELLS = 4_000_000              # 距离矩阵单块元素预算（约 32MB）
COMPARE_CHUNK_CAP = 4096                      # 单块最大行数
COMPARE_PROGRESS_EVERY = 8000                 # 参与比对的图像数超过该值时打印进度
HASH_PROGRESS_EVERY = 2000                    # 哈希计算进度打印阈值（按图像总数）

# ---- 分层切分 ----
BUCKET_MIN_GROUPS = 15    # 轮次池组数达到该值才按框数桶分层（小池分桶会切碎稀有类）
SMALL_POOL_IMAGES = 8    # 池内图像数不超过该值时按"组数"分配，保证 train 一定有组
FEW_SAMPLE_LIMIT = 10    # 类别图像数低于该值时提示补充数据
WARN_NAME_LIMIT = 5      # 报告明细最多列出的文件/类别名个数
LABEL_DECIMALS = 6       # YOLO 标签坐标小数位数


class _SizeMissing(ValueError):
    """XML 缺少有效尺寸且未提供兜底尺寸（调用方可读图像实际尺寸后重试）。"""


def _dw_width(text: str) -> int:
    """计算字符串的终端显示宽度（中日韩字符按 2 列计，其余 1 列）。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def _dw_ljust(text: str, width: int) -> str:
    """按显示宽度左对齐（右侧补空格），中文参与表格时也能对齐。"""
    return text + ' ' * max(0, width - _dw_width(text))


def _dw_rjust(text: str, width: int) -> str:
    """按显示宽度右对齐（左侧补空格）。"""
    return ' ' * max(0, width - _dw_width(text)) + text


def _pct(n: int, denom: int) -> str:
    """格式化 "n (x.x%)"，百分比为 n/denom；分母为 0 时百分比记 0.0%。"""
    return f'{n} ({n / denom:.1%})' if denom else f'{n} (0.0%)'


# ------------------------------------------------------------------ 参数解析
def parse_args(argv=None) -> argparse.Namespace:
    """解析并校验命令行参数。

    Returns:
        解析后的参数对象，额外附带 weights 属性（归一化的 train/val/test 占比）。

    Raises:
        SystemExit: 路径不存在、类别重复/含空白、占比非法、阈值越界时经 argparse 报错退出。
    """
    p = argparse.ArgumentParser(
        prog='split_voc.py',
        description='VOC 数据集 -> YOLO txt 清单式分层切分工具（防泄漏 + 稀有类保障）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--images', type=Path, default=DEFAULT_IMAGES_DIR,
                   help='图像文件夹（不递归子目录）')
    p.add_argument('--anno', '--annotations', dest='anno', type=Path, default=DEFAULT_ANNO_DIR,
                   help='VOC XML 标注文件夹（文件主干与图像一一对应，不递归子目录）')
    p.add_argument('--classes', nargs='+', metavar='NAME', default=list(DEFAULT_CLASSES),
                   help='类别名列表，顺序即 YOLO 类别 id（从 0 开始）')
    p.add_argument('--ratios', nargs='+', type=float, default=list(DEFAULT_RATIOS), metavar='R',
                   help='train/val/test 占比：2 个数（无 test）或 3 个数（test 可为 0），总和约为 1')
    p.add_argument('--output', '-o', type=Path, default=None,
                   help='输出目录；默认 <图像文件夹>_split')
    p.add_argument('--seed', type=int, default=DEFAULT_SEED,
                   help='随机种子，固定后切分结果可复现')
    p.add_argument('--no-dedup', action='store_true',
                   help='跳过 pHash 去重分组（数据为独立照片时用，省时间；连拍/抽帧数据不建议关闭）')
    p.add_argument('--phash-threshold', type=int, default=PHASH_DEFAULT_THRESHOLD, metavar='D',
                   help=f'近似重复判定阈值：汉明距离 <= D 归为同组（0~{PHASH_MAX_DISTANCE}，'
                        f'越大合并越激进；视频抽帧建议 6~10）')
    p.add_argument('--skip-difficult', action='store_true',
                   help='过滤 VOC 中 difficult=1 的目标')
    p.add_argument('--unknown-classes', choices=['error', 'skip'], default='error',
                   help='XML 出现未声明类别时的处理：error=报错退出（默认，防静默漏标），'
                        'skip=跳过该目标')
    p.add_argument('--exclude-background', action='store_true',
                   help='不把无标注文件的图像作为背景图纳入（XML 存在但无有效目标的图仍会作为背景图保留）')
    p.add_argument('--overwrite', action='store_true',
                   help='输出目录已存在清单/data.yaml 时覆盖重建；同时授权重写图像同级'
                        ' labels 目录中已有的标注')
    p.add_argument('--force-split', action='append', nargs=2,
                   metavar=('SPLIT', 'FILE'), default=[],
                   help='把清单文件中的图像强制分到指定子集（组内原子：同组图像跟随'
                        '进入同一子集）；SPLIT ∈ train/val/test，FILE 每行一个文件名'
                        '（带不带扩展名均可，支持 # 注释）。可重复传入多份清单')
    p.add_argument('--force-split-glob', action='append', nargs=2,
                   metavar=('SPLIT', 'PATTERN'), default=[],
                   help='按通配符强制子集（无需清单文件）：图像文件名匹配 PATTERN 的'
                        '整组进入 SPLIT，如 --force-split-glob train "batch0912_*"。'
                        '适合视频帧有固定前缀的场景，可与 --force-split 叠加')
    p.add_argument('--max-group-size', type=int, default=0, metavar='K',
                   help='去重组的最大图像数：连通分量超过 K 张时按相似序拆分'
                        '（长视频抽帧建议 100~300；0 = 不限制）')
    p.add_argument('--copy-out', dest='copy_out', type=str, default=DEFAULT_COPY_OUT,
                   metavar='DIR',
                   help='把自包含训练目录拷贝到指定路径：生成 images/、labels/ 子目录'
                        '与 data.yaml，可整体打包或换机直接训练；不传或为空则仅生成'
                        '清单，不拷贝图像（默认取顶部 DEFAULT_COPY_OUT）')
    args = p.parse_args(argv)

    if not args.images.is_dir():
        p.error(f'图像文件夹不存在: {args.images}（用 --images 指定，或修改脚本配置区默认路径）')
    if not args.anno.is_dir():
        p.error(f'标注文件夹不存在: {args.anno}（用 --anno 指定，或修改脚本配置区默认路径）')
    if len(set(args.classes)) != len(args.classes):
        p.error('--classes 中存在重复类别名')
    if any((not c) or c.strip() != c for c in args.classes):
        p.error('--classes 中存在空白或首尾带空格的类别名')

    # 占比校验与归一化
    r = list(args.ratios)
    if len(r) not in (2, 3):
        p.error('--ratios 需要 2 个（train val）或 3 个数（train val test）')
    if len(r) == 2:
        r.append(0.0)
    if any(x < 0 for x in r):
        p.error('占比不能为负数')
    total = sum(r)
    if total <= 0:
        p.error('占比之和必须大于 0')
    if abs(total - 1.0) > RATIO_SUM_TOLERANCE:
        print(f'[提示] 占比之和为 {total:.3f}，已自动归一化', file=sys.stderr)
    weights = [x / total for x in r]
    if weights[0] <= 0 or weights[1] <= 0:
        p.error('train 与 val 占比必须大于 0（test 允许为 0）')
    args.weights = weights
    if not (0 <= args.phash_threshold <= PHASH_MAX_DISTANCE):
        p.error(f'--phash-threshold 取值范围 0~{PHASH_MAX_DISTANCE}')
    for split_name, list_path in args.force_split:
        if split_name not in SPLIT_NAMES:
            p.error(f'--force-split 子集名必须是 {" / ".join(SPLIT_NAMES)}，收到: {split_name}')
        if not Path(list_path).is_file():
            p.error(f'--force-split 清单文件不存在: {list_path}')
    for split_name, _pattern in args.force_split_glob:
        if split_name not in SPLIT_NAMES:
            p.error(f'--force-split-glob 子集名必须是 {" / ".join(SPLIT_NAMES)}，'
                    f'收到: {split_name}')
    if args.max_group_size < 0:
        p.error('--max-group-size 不能为负（0 表示不限制）')
    if args.copy_out is not None and not str(args.copy_out).strip():
        args.copy_out = None        # 传空字符串视为未传
    return args


# ------------------------------------------------------------------ 文件扫描
def scan_files(directory: Path, exts) -> Dict[str, Path]:
    """扫描目录（不递归），返回 {文件主干(小写): 路径}。

    主干重名（如 A.jpg 与 A.png 并存）时保留排序靠前者并警告，
    避免标注匹配出现二义性。
    """
    mapping: Dict[str, Path] = {}
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() in exts:
            key = f.stem.lower()
            if key in mapping:
                print(f'[警告] 文件主干重名，忽略后者: {mapping[key].name} / {f.name}',
                      file=sys.stderr)
                continue
            mapping[key] = f
    return mapping


# ------------------------------------------------------------------ 图像探测
def _phash_from_gray(a) -> int:
    """32x32 灰度矩阵 -> 64 位 pHash。

    DCT-II -> 左上 8x8 低频系数按中值二值化。直流分量不参与中值、
    对应位恒 0，使哈希对整体亮度变化不敏感。

    Args:
        a: 形状 (32, 32) 的 float64 灰度矩阵。

    Returns:
        64 位整数哈希值。
    """
    import numpy as np
    n = PHASH_IMG_SIZE
    idx = np.arange(n)
    # DCT-II 正交基矩阵：dct[k, m] = sqrt(2/n)·cos(pi·(2m+1)·k / (2n))，第 0 行除 sqrt(2)
    dct = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * idx[None, :] + 1) * idx[:, None] / (2 * n))
    dct[0, :] = np.sqrt(1.0 / n)
    freq = dct @ a @ dct.T
    low = freq[:PHASH_LOW_FREQ, :PHASH_LOW_FREQ].flatten()
    bits = low > np.median(low[1:])   # 中值不含直流分量
    bits[0] = False                   # 直流位恒 0
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return int(v)


def probe_image(img_path: Path, need_hash: bool):
    """打开图像一次，返回 (pHash 或 None, (宽, 高) 或 None, 失败原因 或 None)。

    need_hash=False 时只读文件头取尺寸，不解码像素（快）；
    计算哈希才完整解码。Windows 下刚复制/被杀软扫描的文件可能瞬时
    占用，对 OSError 做短重试；其他异常不重试。

    任何一步失败时保留已拿到的部分（如尺寸可用而哈希失败），其余为 None。
    """
    try:
        from PIL import Image
    except ImportError:
        return None, None, '缺少 pillow（pip install pillow）'
    np = None
    if need_hash:
        try:
            import numpy as np
        except ImportError:
            return None, None, '缺少 numpy（pip install numpy）'

    phash_val, wh, err = None, None, None
    for attempt in range(HASH_RETRY_TIMES):
        try:
            with Image.open(str(img_path)) as im:
                wh = im.size                      # (宽, 高)，惰性读取不解码
                if need_hash:
                    gray = im.convert('L').resize(
                        (PHASH_IMG_SIZE, PHASH_IMG_SIZE), Image.LANCZOS)
                    phash_val = _phash_from_gray(np.asarray(gray, dtype=np.float64))
            return phash_val, wh, None            # 成功：错误信息清空
        except OSError as e:
            # 占用/截断等 IO 类错误：短暂等待后重试
            err = f'{type(e).__name__}: {e}'
            time.sleep(HASH_RETRY_SLEEP * (attempt + 1))
        except Exception as e:
            # 解码库的其他异常（如分辨率炸弹）：重试无意义
            return None, wh, f'{type(e).__name__}: {e}'
    return phash_val, wh, err


# ------------------------------------------------------------------ 场景分组
class _UnionFind:
    """轻量并查集（路径压缩 + 可选组大小上限），用于求相似图像的连通分量。"""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n                     # 各集合当前大小

    def find(self, x: int) -> int:
        """返回 x 所在集合的根节点，顺带路径压缩。"""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int, max_size: int = 0) -> bool:
        """合并两个集合，返回是否发生了合并。

        max_size > 0 时，合并后集合大小超过上限则拒绝——用于
        --max-group-size 拆分长视频传递链形成的巨组。
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if max_size and self.size[ra] + self.size[rb] > max_size:
            return False
        if self.size[ra] < self.size[rb]:       # 小树挂大树，保持扁平
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def build_scene_groups(n_records: int, hashes: List[Optional[int]], threshold: int,
                       max_group_size: int = 0):
    """按 pHash 相似度把记录聚成组（并查集连通分量）。

    哈希为 None（解码失败）的记录不参与比对、自成一组，避免全 0
    哈希互相误合并。组内图像切分时视为原子单位（永远进同一子集），
    防止同场景图像同时出现在 train 与 val/test 造成指标虚高。

    Args:
        n_records: 记录总数。
        hashes: 与记录对齐的哈希列表，无哈希处为 None。
        threshold: 汉明距离阈值，<= 该值判为近似重复。
        max_group_size: 组大小上限（>0 时超限合并被拒绝并计数），
            用于拆分长视频传递链形成的巨组。

    Returns:
        (组列表, 合并边数, 因大小上限被拒绝的合并对数)。
        组列表按首成员索引排序，保证确定性。
    """
    import numpy as np
    ok = [i for i, h in enumerate(hashes) if h is not None]
    uf = _UnionFind(n_records)
    merged = 0
    capped = 0
    m = len(ok)
    if m >= 2:
        hv = np.array([hashes[i] for i in ok], dtype=np.uint64)
        # 0~255 每个字节的置位数查表：uint64 汉明距离 = 8 个字节置位数之和
        lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
        # 分块两两比对：块大小由内存预算决定，避免大矩阵一次成型
        chunk = max(1, min(COMPARE_CHUNK_CAP, COMPARE_BUDGET_CELLS // m))
        for start in range(0, m, chunk):
            block = hv[start:start + chunk]
            xor = block[:, None] ^ hv[None, :]                 # (块, m) uint64
            codes = xor.view(np.uint8).reshape(len(block), -1, 8)
            dist = lut[codes].sum(axis=2)                      # 汉明距离矩阵
            for li, lj in np.argwhere(dist <= threshold):
                i, j = ok[start + int(li)], ok[int(lj)]
                # 只处理 i < j 的对角线上方，避免重复合并
                if i < j and uf.find(i) != uf.find(j):
                    if uf.union(i, j, max_group_size):
                        merged += 1
                    else:
                        capped += 1    # 因 --max-group-size 被拒绝的合并
            if m > COMPARE_PROGRESS_EVERY:
                print(f'  比对进度 {min(start + len(block), m)}/{m}')
    members = defaultdict(list)
    for i in range(n_records):
        members[uf.find(i)].append(i)
    groups = sorted(members.values(), key=lambda g: g[0])      # 确定性顺序
    return groups, merged, capped


# ------------------------------------------------------------------ VOC 解析
def parse_voc_xml(xml_path: Path, class_ids: Dict[str, int],
                  fallback_size: Optional[Tuple[int, int]],
                  skip_difficult: bool, unknown: Counter):
    """解析单个 VOC XML。

    Args:
        xml_path: XML 文件路径。
        class_ids: 类别名 -> id 映射。
        fallback_size: XML 尺寸缺失/非法时使用的兜底尺寸 (宽, 高)，可为 None。
        skip_difficult: 是否过滤 difficult=1 的目标。
        unknown: 未声明类别计数器（跨文件累计）。

    Returns:
        (YOLO 标签行列表, 出现的类别 id 集合, 每类框数 Counter, 本文件统计)。
        统计键: difficult / unknown / badbox / clipped / size_fallback。

    Raises:
        _SizeMissing: XML 缺少有效尺寸且未提供兜底尺寸。
        ValueError: XML 语法错误无法解析。
    """
    try:
        root = ET.parse(str(xml_path)).getroot()
    except ET.ParseError as e:
        raise ValueError(f'XML 语法错误: {e}') from e
    size = root.find('size')
    w_txt = size.findtext('width') if size is not None else None
    h_txt = size.findtext('height') if size is not None else None
    try:
        w = int(round(float(w_txt))) if w_txt else 0
        h = int(round(float(h_txt))) if h_txt else 0
    except ValueError:
        w = h = 0

    st: Counter = Counter()
    if w <= 0 or h <= 0:
        if fallback_size and fallback_size[0] > 0 and fallback_size[1] > 0:
            # XML 尺寸缺失/非法 -> 用图像实际尺寸兜底（不丢标注）
            w, h = fallback_size
            st['size_fallback'] = 1
        else:
            raise _SizeMissing('XML 缺少有效尺寸，且未提供兜底尺寸')

    lines: List[str] = []
    classes = set()
    per_class: Counter = Counter()
    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip()
        try:
            difficult = int(float(obj.findtext('difficult') or 0))
        except ValueError:
            difficult = 0
        if skip_difficult and difficult:
            st['difficult'] += 1
            continue
        if name not in class_ids:
            st['unknown'] += 1
            unknown[name] += 1
            continue
        bb = obj.find('bndbox')
        if bb is None:
            st['badbox'] += 1
            continue
        try:
            x0 = float(bb.findtext('xmin'))
            y0 = float(bb.findtext('ymin'))
            x1 = float(bb.findtext('xmax'))
            y1 = float(bb.findtext('ymax'))
        except (TypeError, ValueError):
            st['badbox'] += 1
            continue
        # 越界坐标裁剪到图像范围内
        cx0, cy0 = min(max(x0, 0.0), float(w)), min(max(y0, 0.0), float(h))
        cx1, cy1 = min(max(x1, 0.0), float(w)), min(max(y1, 0.0), float(h))
        if not (cx1 > cx0 and cy1 > cy0):
            st['badbox'] += 1
            continue
        if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
            st['clipped'] += 1
        cid = class_ids[name]
        # VOC 像素坐标 -> YOLO 归一化中心点格式
        xc = (cx0 + cx1) / 2.0 / w
        yc = (cy0 + cy1) / 2.0 / h
        bw = (cx1 - cx0) / w
        bh = (cy1 - cy0) / h
        lines.append(f'{cid} {xc:.{LABEL_DECIMALS}f} {yc:.{LABEL_DECIMALS}f} '
                     f'{bw:.{LABEL_DECIMALS}f} {bh:.{LABEL_DECIMALS}f}')
        classes.add(cid)
        per_class[cid] += 1
    return lines, classes, per_class, st


def collect_records(img_map: Dict[str, Path], xml_map: Dict[str, Path],
                    class_ids: Dict[str, int], skip_difficult: bool,
                    dedup: bool, include_background: bool):
    """匹配图像与 XML 并解析为记录列表。

    Args:
        img_map: {主干: 图像路径}。
        xml_map: {主干: XML 路径}。
        class_ids: 类别名 -> id 映射。
        skip_difficult: 是否过滤 difficult 目标。
        dedup: 是否开启去重（需全量计算 pHash）。
        include_background: 无标注图像是否按背景图纳入。

    Returns:
        (records, stats, unknown, parse_fail, probe_fail)。
        record 字段: stem / path / hash / lines / classes / per_class / is_bg。
    """
    records: List[dict] = []
    stats: Counter = Counter()
    unknown: Counter = Counter()
    parse_fail: List[str] = []      # XML 解析失败明细
    probe_fail: List[str] = []      # 图像读取/解码异常明细
    size_cache: Dict[str, Tuple[int, int]] = {}

    # ---- 去重探测：全量读图一次（哈希 + 尺寸），尺寸缓存供 XML 兜底复用 ----
    hashes: Dict[str, Optional[int]] = {}
    if dedup:
        n_total = len(img_map)
        print(f'[去重] 计算 {n_total} 张图像的 pHash ...')
        for k, stem in enumerate(sorted(img_map), 1):
            h, wh, err = probe_image(img_map[stem], need_hash=True)
            if wh is None:
                # 连尺寸都读不到：训练时也无法解码，直接排除并报告
                probe_fail.append(f'{img_map[stem].name}（{err or "无法读取"}，已排除）')
                continue
            if h is None:
                probe_fail.append(f'{img_map[stem].name}（{err or "哈希失败"}，按独立组处理）')
            hashes[stem] = h
            size_cache[stem] = wh
            if n_total > HASH_PROGRESS_EVERY and k % 1000 == 0:
                print(f'  已计算 {k}/{n_total}')

    for stem in sorted(img_map):
        if dedup and stem not in size_cache:
            continue                 # 探测阶段已判死的图
        img_path = img_map[stem]
        xml_path = xml_map.get(stem)
        lines: List[str] = []
        classes = set()
        per_class: Counter = Counter()

        if xml_path is None:
            # 无标注文件：按背景图纳入或排除
            if not include_background:
                stats['no_xml_excluded'] += 1
                continue
            stats['no_xml_bg'] += 1
        else:
            fb = size_cache.get(stem)
            try:
                lines, classes, per_class, st = parse_voc_xml(
                    xml_path, class_ids, fb, skip_difficult, unknown)
                stats.update(st)
            except _SizeMissing:
                # XML 尺寸缺失且无缓存 -> 惰性读一次图像尺寸再重试
                wh = probe_image(img_path, need_hash=False)[1]
                if wh is None:
                    parse_fail.append(f'{xml_path.name}（尺寸缺失，且图像无法读取兜底）')
                    continue
                size_cache[stem] = wh
                lines, classes, per_class, st = parse_voc_xml(
                    xml_path, class_ids, wh, skip_difficult, unknown)
                stats.update(st)
            except ValueError as e:
                parse_fail.append(f'{xml_path.name}（{e}）')
                continue
            if not classes:
                stats['empty_xml_bg'] += 1   # XML 存在但无有效目标 -> 背景图

        records.append({
            'stem': stem, 'path': img_path, 'hash': hashes.get(stem),
            'lines': lines, 'classes': classes, 'per_class': per_class,
            'is_bg': not classes,
        })
    return records, stats, unknown, parse_fail, probe_fail


# ------------------------------------------------------------------ 分层切分
def alloc_counts(n: int, weights) -> List[int]:
    """最大余数法把 n 个样本按 weights 比例分配为各桶数量。

    兜底规则（优先级从高到低，捐献方剩余须 >= 2）：
      1. train 至少 1 个（n >= 1）；
      2. val 至少 1 个（该桶占比 > 0 且 n >= 2）；
      3. test 至少 1 个（该桶占比 > 0 且 n >= 3）。

    Args:
        n: 样本总数，须 >= 1。
        weights: 各桶占比列表（train, val, test 顺序），和为 1。

    Returns:
        各桶数量列表，总和恰为 n。
    """
    k = len(weights)
    quotas = [n * w for w in weights]
    counts = [int(q) for q in quotas]
    rem = n - sum(counts)
    # 按小数余量从大到小补 1，平局时占比大的优先（结果确定）
    order = sorted(range(k), key=lambda i: (quotas[i] - counts[i], weights[i]),
                   reverse=True)
    for i in order[:rem]:
        counts[i] += 1

    def top_up(idx: int, min_n: int) -> None:
        if idx >= k:
            return
        if weights[idx] > 0 and counts[idx] == 0 and n >= min_n:
            donors = [j for j in range(k) if j != idx and counts[j] >= 2]
            if donors:
                counts[max(donors, key=lambda j: counts[j])] -= 1
                counts[idx] += 1

    top_up(0, 1)   # train
    top_up(1, 2)   # val
    top_up(2, 3)   # test
    return counts


def assign_splits(records: List[dict], groups: List[List[int]], weights,
                  rng: random.Random, class_names: List[str],
                  forced: Optional[Dict[int, str]] = None) -> List[str]:
    """分组 + 稀有类优先 + 框数桶分层的切分。

    Args:
        records: 记录列表（collect_records 的返回值）。
        groups: 每组为记录索引列表（去重组或单图组），切分的原子单位。
        weights: 归一化的 (train, val, test) 占比。
        rng: 已定种子的随机数生成器。
        class_names: 类别名列表（仅用于日志打印）。
        forced: {组索引: 子集名}，用户强制指定的组——预置进结果、
            不参与任何池分配与兜底挪动（组内原子优先级最高）。

    Returns:
        与 records 对齐的子集名列表（'train' / 'val' / 'test'）。

    三层策略叠加（两版旧脚本的优点合并）：
      1) 稀有类优先轮次：组归入"组内最稀有类"的轮次，轮次按类别组数
         升序处理，稀有类先分配，val/test 覆盖有保障；
      2) 框数桶分层：轮次池够大时按组内总框数 0/1/2/3+ 分桶，各桶独立
         按配额装填，保持各子集的目标密度分布一致；
      3) 小池按组分配 / 大池按图配额：小池用最大余数法按"组数"分配，
         保证 train 一定有组（稀有类的生命线）；大池对 val/test 设图像
         配额上限、装不下的默认进 train，并对大组做容差控制，防止
         连拍大组独占 val/test 配额。
    每轮结束兜底检查：本类 val/test 仍无组时从 train 挪最小的组补上。
    """
    r_train, r_val, r_test = weights

    # ---- 组级特征：类别并集 / 总框数 / 图像数 ----
    g_classes, g_boxes, g_size = [], [], []
    for g in groups:
        cs = set()
        boxes = 0
        for i in g:
            cs |= records[i]['classes']
            boxes += sum(records[i]['per_class'].values())
        g_classes.append(cs)
        g_boxes.append(boxes)
        g_size.append(len(g))

    # 每类别的组数（决定轮次顺序：稀有类先处理）
    cls_groups = defaultdict(list)
    for gi, cs in enumerate(g_classes):
        for c in cs:
            cls_groups[c].append(gi)
    cls_n = {c: len(v) for c, v in cls_groups.items()}

    def round_of(cs):
        """组 -> 轮次类别：取组内'最稀有'类（平局按类别 id，保证确定性）。"""
        if not cs:
            return None
        return min(cs, key=lambda c: (cls_n[c], c))

    g_round = [round_of(cs) for cs in g_classes]
    # 轮次顺序：类别按组数升序，背景轮（None）最后
    order = sorted(cls_n, key=lambda c: (cls_n[c], c)) + [None]
    # 强制组预置进结果：后续所有池分配与兜底逻辑都不会再碰它们
    split_of_group: Dict[int, str] = dict(forced or {})

    def assign_pool(pool: List[int]) -> None:
        """把一个池（已洗牌）按策略装填到三个子集。"""
        total_imgs = sum(g_size[gi] for gi in pool)
        if total_imgs <= SMALL_POOL_IMAGES:
            # 小池：按组数最大余数分配，train 优先补余，保证 train 一定有组
            counts = alloc_counts(len(pool), (r_train, r_val, r_test))
            seq = (['train'] * counts[0] + ['val'] * counts[1]
                   + ['test'] * counts[2])
            for gi, place in zip(pool, seq):
                split_of_group[gi] = place
            return
        # 大池：val/test 设图像配额上限，装不下的默认进 train
        caps = {'val': round(total_imgs * r_val),
                'test': round(total_imgs * r_test) if r_test > 0 else 0}
        # 容差：允许小幅超配，让"组大小略超配额"时 val/test 不至于分不到；
        # 上限取配额一半，避免大组把小配额掏空
        tols = {s: max(2, q // 2) for s, q in caps.items()}
        got = {'val': 0, 'test': 0}
        for gi in pool:
            sz = g_size[gi]
            place = 'train'
            for s in ('val', 'test'):
                if got[s] < caps[s] and got[s] + sz <= caps[s] + tols[s]:
                    place = s
                    break
            if place != 'train':
                got[place] += sz
            split_of_group[gi] = place

    def covered(c: int, target: str) -> bool:
        """类别 c 当前是否已有组落在 target 子集。"""
        return any(split_of_group.get(gi) == target for gi in cls_groups[c])

    for r in order:
        pool = [gi for gi in range(len(groups))
                if g_round[gi] == r and gi not in split_of_group]
        if not pool:
            continue
        rng.shuffle(pool)
        pool_imgs = sum(g_size[gi] for gi in pool)
        # 池足够大才按框数桶分层；小池分桶会把稀有类切碎
        bucketed = r is not None and len(pool) >= BUCKET_MIN_GROUPS
        if bucketed:
            by_bucket = defaultdict(list)
            for gi in pool:
                by_bucket[min(g_boxes[gi], 3)].append(gi)
            subpools = [by_bucket[b] for b in sorted(by_bucket)]
        else:
            subpools = [pool]
        label = class_names[r] if r is not None else '背景'
        print(f'  轮次 {label}: {len(pool)} 组 / {pool_imgs} 图'
              f'{"，按框数桶分层" if bucketed else ""}')
        for sub in subpools:
            assign_pool(sub)
        # 兜底：本类 val/test 仍无组时，从 train 挪最小的组补上（不碰强制组）
        if r is not None:
            targets = [('val', 2)] + ([('test', 3)] if r_test > 0 else [])
            for target, min_groups in targets:
                if cls_n[r] >= min_groups and not covered(r, target):
                    cands = [gi for gi in cls_groups[r]
                             if gi not in forced
                             and split_of_group.get(gi) == 'train']
                    if cands:
                        gi = min(cands, key=lambda g: (g_size[g], g))
                        split_of_group[gi] = target

    # 全局兜底：极端情况下 val/test 可能为空（如纯背景数据集）
    for target, min_n in [('val', 2)] + ([('test', 3)] if r_test > 0 else []):
        if len(records) >= min_n and not any(v == target for v in split_of_group.values()):
            cands = [gi for gi, v in split_of_group.items()
                     if gi not in forced and v == 'train']
            if cands:
                gi = min(cands, key=lambda g: (g_size[g], g))
                split_of_group[gi] = target

    # 展开到记录级：组内所有记录继承组的子集
    split_of: List[str] = [None] * len(records)
    for gi, g in enumerate(groups):
        for i in g:
            split_of[i] = split_of_group[gi]
    return split_of


# ------------------------------------------------------------------ 输出落盘
def write_outputs(out: Path, labels_dir: Path, records: List[dict],
                  split_of: List[str], classes: List[str],
                  splits_active: Tuple[str, ...]) -> None:
    """写出 YOLO 标注（总是重新生成）、txt 清单与 data.yaml。

    labels 固定生成在 images 同级镜像目录：Ultralytics 训练时按
    "图像路径中的 images 段替换为 labels" 定位标注。
    """
    # 1) labels：总是重写，保证与当前 --classes 类别表一致
    labels_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        (labels_dir / (rec['stem'] + '.txt')).write_text(
            ''.join(line + '\n' for line in rec['lines']), encoding='utf-8')

    # 2) txt 清单：每行一张图的绝对路径（Ultralytics 清单训练方式）
    out.mkdir(parents=True, exist_ok=True)
    for s in splits_active:
        paths = [str(rec['path'].resolve())
                 for rec, sp in zip(records, split_of) if sp == s]
        (out / f'{s}.txt').write_text(
            '\n'.join(paths) + ('\n' if paths else ''), encoding='utf-8')
    # 本次未启用的子集若残留旧清单则删除，避免误用
    for s in SPLIT_NAMES:
        if s not in splits_active:
            f = out / f'{s}.txt'
            if f.is_file():
                f.unlink()

    # 3) data.yaml：路径统一 posix 正斜杠并加引号，避免反斜杠转义问题
    lines = ['# 自动生成：VOC -> YOLO 清单式数据集配置（YOLOv8/v11/26 通用）',
             f'path: "{out.resolve().as_posix()}"']
    for s in splits_active:
        lines.append(f'{s}: {s}.txt')
    lines.append('names:')
    for i, c in enumerate(classes):
        lines.append(f'  {i}: "{c}"')
    (out / 'data.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ------------------------------------------------------------------ 自包含拷贝
def write_copyout_dataset(copy_root: Path, records: List[dict],
                          split_of: List[str], classes: List[str],
                          splits_active: Tuple[str, ...], overwrite: bool) -> None:
    """生成自包含 YOLO 训练目录（--copy-out）。

    结构（目录式数据集，与源数据完全解耦，可直接打包/换机训练）：
      <copy_root>/images/<split>/<原图>
      <copy_root>/labels/<split>/<主干>.txt
      <copy_root>/data.yaml
    """
    dirty = any((copy_root / sub).is_dir() and any((copy_root / sub).iterdir())
                for sub in ('images', 'labels')) \
        or (copy_root / 'data.yaml').is_file()
    if dirty and not overwrite:
        sys.exit(f'[错误] 拷贝目标目录已有内容: {copy_root}\n'
                 f'       确认覆盖请加 --overwrite 重新运行。')
    if dirty:
        for sub in ('images', 'labels'):
            if (copy_root / sub).is_dir():
                shutil.rmtree(copy_root / sub)
        if (copy_root / 'data.yaml').is_file():
            (copy_root / 'data.yaml').unlink()

    for s in splits_active:
        (copy_root / 'images' / s).mkdir(parents=True, exist_ok=True)
        (copy_root / 'labels' / s).mkdir(parents=True, exist_ok=True)
    for rec, sp in zip(records, split_of):
        # 复制图像；标签按当前 --classes 重新生成（不用源 labels，保证一致性）
        shutil.copy2(rec['path'], copy_root / 'images' / sp / rec['path'].name)
        (copy_root / 'labels' / sp / (rec['stem'] + '.txt')).write_text(
            ''.join(line + '\n' for line in rec['lines']), encoding='utf-8')

    lines = ['# 自动生成：自包含 YOLO 训练目录（Ultralytics 通用）',
             f'path: "{copy_root.resolve().as_posix()}"']
    for s in splits_active:
        lines.append(f'{s}: images/{s}')
    lines.append('names:')
    for i, c in enumerate(classes):
        lines.append(f'  {i}: "{c}"')
    (copy_root / 'data.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')


# ------------------------------------------------------------------ 结果报告
def print_report(args, records: List[dict], split_of: List[str],
                 groups: List[List[int]], merged: int, stats: Counter,
                 unknown: Counter, parse_fail: List[str], probe_fail: List[str],
                 splits_active: Tuple[str, ...], labels_dir: Path,
                 forced: Dict[int, str], capped: int,
                 copy_root: Optional[Path] = None) -> None:
    """打印切分结果统计表与异常提示。"""
    total = len(records)
    box_total = sum(len(r['lines']) for r in records)
    counts = {s: sum(1 for v in split_of if v == s) for s in splits_active}
    print('=' * 66)
    print('数据集切分完成')
    print('=' * 66)
    print(f'输出目录 : {args.output}')
    print(f'YOLO标注 : {labels_dir}（images 同级 labels，Ultralytics 约定）')
    print(f'随机种子 : {args.seed}（对比实验请保持一致）')
    print(f'图像总数 : {total} 张，目标框共 {box_total} 个')
    print('切分数量 : ' + '  '.join(
        f'{s}={counts[s]}({counts[s] / total:.1%})' for s in splits_active))
    n_bg = sum(1 for r in records if r['is_bg'])
    if n_bg:
        print(f'背景图   : {n_bg} 张（空标签，参与训练可抑制误检）')
    if args.no_dedup:
        print('去重分组 : 已跳过（--no-dedup；连拍/抽帧数据不建议关闭）')
    else:
        sizes = sorted((len(g) for g in groups), reverse=True)
        multi = sum(1 for s in sizes if s > 1)
        print(f'去重分组 : {len(groups)} 组（多图组 {multi} 个，最大组 {sizes[0]} 张，'
              f'合并 {merged} 对）—— 组内图像固定进同一子集，防泄漏')
        if capped:
            print(f'           因 --max-group-size={args.max_group_size} 拒绝合并 '
                  f'{capped} 对（限制最大组规模，防止整段视频绑成一个组）')
    if forced:
        parts = []
        for s in splits_active:
            gs = [gi for gi, sp in forced.items() if sp == s]
            if gs:
                parts.append(f'{s}={len(gs)}组/{sum(len(groups[gi]) for gi in gs)}图')
        print('强制指定 : ' + ('  '.join(parts) if parts else '-')
              + '（--force-split，组内原子生效）')
    if copy_root:
        print(f'自包含目录 : {copy_root}（图像+标注已拷贝，可整体打包/换机训练）')

    print('-' * 66)
    print('表 1 / 各类别【图像】统计：格子 = 含该类的图像张数；占比行 = 占图像总数')
    w = 14

    def _pctf(n: int, denom: int) -> str:
        """裸百分比字符串（不带数量），分母为 0 时记 0.0%。"""
        return f'{n / denom:.1%}' if denom else '0.0%'

    print(f'{"id":>3}  ' + _dw_ljust('类别', 16)
          + ''.join(_dw_rjust(s, w) for s in splits_active)
          + _dw_rjust('合计', w) + '   备注')
    for cid, cname in enumerate(args.classes):
        imgs = {s: sum(1 for r, v in zip(records, split_of)
                       if v == s and cid in r['classes']) for s in splits_active}
        tot = sum(imgs.values())
        notes = []
        if imgs.get('val', 1) == 0:
            notes.append('val 无样本!')
        if 'test' in imgs and imgs['test'] == 0:
            notes.append('test 无样本')
        if tot < FEW_SAMPLE_LIMIT:
            notes.append(f'样本过少(<{FEW_SAMPLE_LIMIT})，建议补充')
        print(f'{cid:>3}  ' + _dw_ljust(cname, 16)
              + ''.join(str(imgs[s]).rjust(w) for s in splits_active)
              + str(tot).rjust(w)
              + '   ' + ('[' + '; '.join(notes) + ']' if notes else ''))
        print('     ' + _dw_ljust('占比', 16)
              + ''.join(_pctf(imgs[s], total).rjust(w) for s in splits_active)
              + _pctf(tot, total).rjust(w))
    # 背景图（空标签）在各子集中的分布行
    if n_bg:
        bg = {s: sum(1 for r, v in zip(records, split_of)
                     if v == s and r['is_bg']) for s in splits_active}
        print(f'{"bg":>3}  ' + _dw_ljust('背景图(空标签)', 16)
              + ''.join(str(bg[s]).rjust(w) for s in splits_active)
              + str(n_bg).rjust(w))
        print('     ' + _dw_ljust('占比', 16)
              + ''.join(_pctf(bg[s], total).rjust(w) for s in splits_active)
              + _pctf(n_bg, total).rjust(w))

    print('-' * 66)
    print('表 2 / 各类别【框】统计：格子 = 该类的目标框数；占比行 = 占框总数')
    print(f'{"id":>3}  ' + _dw_ljust('类别', 16)
          + ''.join(_dw_rjust(s, w) for s in splits_active)
          + _dw_rjust('合计', w))
    for cid, cname in enumerate(args.classes):
        boxes = {s: sum(r['per_class'].get(cid, 0)
                        for r, v in zip(records, split_of) if v == s)
                 for s in splits_active}
        btot = sum(boxes.values())
        print(f'{cid:>3}  ' + _dw_ljust(cname, 16)
              + ''.join(str(boxes[s]).rjust(w) for s in splits_active)
              + str(btot).rjust(w))
        print('     ' + _dw_ljust('占比', 16)
              + ''.join(_pctf(boxes[s], box_total).rjust(w) for s in splits_active)
              + _pctf(btot, box_total).rjust(w))

    print('-' * 66)
    msgs = []
    if stats['xml_no_img']:
        msgs.append(f"{stats['xml_no_img']} 个 XML 无对应图像，已忽略")
    if stats['no_xml_bg']:
        msgs.append(f"{stats['no_xml_bg']} 张图像无标注文件，按背景图纳入（空标签）")
    if stats['no_xml_excluded']:
        msgs.append(f"{stats['no_xml_excluded']} 张图像无标注文件，未纳入（--exclude-background）")
    if stats['empty_xml_bg']:
        msgs.append(f"{stats['empty_xml_bg']} 个 XML 无有效目标，按背景图处理")
    if stats['size_fallback']:
        msgs.append(f"{stats['size_fallback']} 个 XML 尺寸缺失/非法，已用图像实际尺寸兜底")
    if parse_fail:
        show = '; '.join(parse_fail[:WARN_NAME_LIMIT]) \
            + ('...' if len(parse_fail) > WARN_NAME_LIMIT else '')
        msgs.append(f'{len(parse_fail)} 个 XML 解析失败被排除: {show}')
    if probe_fail:
        show = '; '.join(probe_fail[:WARN_NAME_LIMIT]) \
            + ('...' if len(probe_fail) > WARN_NAME_LIMIT else '')
        msgs.append(f'{len(probe_fail)} 张图像读取/解码异常: {show}')
    if stats['difficult']:
        msgs.append(f"跳过 difficult=1 目标 {stats['difficult']} 个")
    if stats['badbox']:
        msgs.append(f'跳过退化/无效目标框 {stats["badbox"]} 个')
    if stats['clipped']:
        msgs.append(f'裁剪越界目标框 {stats["clipped"]} 个')
    if unknown and args.unknown_classes == 'skip':
        names_show = ', '.join(list(unknown)[:WARN_NAME_LIMIT])
        msgs.append(f"跳过未声明类别目标 {sum(unknown.values())} 个（{names_show}）")
    if msgs:
        for m in msgs:
            print(f'[提示] {m}')
    else:
        print('[提示] 未发现异常')
    print('=' * 66)
    yaml_path = ((copy_root or args.output) / 'data.yaml').as_posix()
    print('下一步（模型可换 yolo26s/m/l 或 v8/v11 系列）:')
    print(f'  yolo detect train model=yolo26n.pt data="{yaml_path}" epochs=100 imgsz=640')


# ------------------------------------------------------------------ 主流程
def main(argv=None) -> None:
    """主流程：扫描 -> 解析 -> 分组 -> 切分 -> 落盘 -> 报告。"""
    args = parse_args(argv)
    class_ids = {n: i for i, n in enumerate(args.classes)}
    img_dir = args.images.resolve()
    ann_dir = args.anno.resolve()
    out = args.output.resolve() if args.output else img_dir.parent / (img_dir.name + '_split')
    args.output = out
    labels_dir = img_dir.parent / 'labels'

    # Ultralytics 依"路径中的 images 段替换为 labels"定位标注，目录名不合约定时提醒
    if '/images/' not in (img_dir.as_posix() + '/'):
        print('[警告] 图像文件夹路径中不含 "images" 目录段，Ultralytics 可能找不到'
              '同级 labels 标注，建议把图像放在名为 images 的文件夹内', file=sys.stderr)

    # 覆盖保护：输出目录已有本脚本产物时，必须显式 --overwrite
    existing = [out / f'{s}.txt' for s in SPLIT_NAMES if (out / f'{s}.txt').is_file()]
    if (out / 'data.yaml').is_file():
        existing.append(out / 'data.yaml')
    if existing and not args.overwrite:
        sys.exit(f'[错误] 输出目录已有清单/data.yaml: {out}\n'
                 f'       确认覆盖重建请加 --overwrite 重新运行。')

    # 覆盖保护（labels）：标注写在图像同级 labels 目录，重跑会按当前
    # --classes 全量重新生成；已存在时必须显式 --overwrite，防止静默篡改
    label_existing = list(labels_dir.glob('*.txt')) if labels_dir.is_dir() else []
    if label_existing and not args.overwrite:
        sys.exit(f'[错误] 已存在 YOLO 标注目录: {labels_dir}（{len(label_existing)} 个 txt）\n'
                 f'       重跑将按当前类别表重新生成全部标注；确认覆盖请加 --overwrite。')

    # 去重功能的依赖提前拦截，避免逐图重复报错
    if not args.no_dedup:
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as e:
            sys.exit(f'[错误] 去重功能需要 pillow 和 numpy（缺少 {e.name}）。\n'
                     f'       安装: pip install pillow numpy；或用 --no-dedup 关闭去重。')

    print(f'[扫描] 图像目录: {img_dir}')
    print(f'[扫描] 标注目录: {ann_dir}')
    img_map = scan_files(img_dir, IMG_EXTS)
    xml_map = scan_files(ann_dir, {'.xml'})
    if not img_map:
        sys.exit(f'[错误] 图像文件夹中没有找到图像: {img_dir}')
    if not xml_map:
        sys.exit(f'[错误] 标注文件夹中没有找到 XML: {ann_dir}')

    records, stats, unknown, parse_fail, probe_fail = collect_records(
        img_map, xml_map, class_ids, args.skip_difficult,
        dedup=not args.no_dedup, include_background=not args.exclude_background)
    stats['xml_no_img'] = len(set(xml_map) - set(img_map))

    # 未知类别默认报错退出：宁可失败也不静默漏标/错标
    if unknown and args.unknown_classes == 'error':
        print('[错误] XML 中出现未在 --classes 声明的类别名:', file=sys.stderr)
        for name, cnt in unknown.most_common():
            print(f'       {name}（{cnt} 个目标）', file=sys.stderr)
        print('       请将其加入 --classes，或用 --unknown-classes skip 跳过这些目标。',
              file=sys.stderr)
        sys.exit(1)
    if not records:
        sys.exit('[错误] 没有可用数据，请检查图像与 XML 文件名是否一一对应。')

    # ---- 读取 --force-split 清单：主干 -> 子集（未知主干警告并忽略）----
    force_of_stem: Dict[str, str] = {}
    for split_name, list_path in args.force_split:
        with open(list_path, encoding='utf-8-sig') as fh:
            stems = [ln.strip() for ln in fh
                     if ln.strip() and not ln.strip().startswith('#')]
        miss = [s for s in stems if Path(s).stem.lower() not in img_map]
        if miss:
            show = ', '.join(miss[:WARN_NAME_LIMIT]) + \
                ('...' if len(miss) > WARN_NAME_LIMIT else '')
            print(f'[警告] --force-split {split_name}: {len(miss)} 个文件名不在数据集中，'
                  f'已忽略: {show}', file=sys.stderr)
        for s in stems:
            key = Path(s).stem.lower()      # 兼容清单里带/不带扩展名
            if key not in img_map:
                continue
            prev = force_of_stem.get(key)
            if prev and prev != split_name:
                sys.exit(f'[错误] 同一图像出现在多份强制清单且目标冲突: {key}'
                         f'（{prev} / {split_name}）')
            force_of_stem[key] = split_name

    # ---- 读取 --force-split-glob：按文件名通配符圈定（无需清单文件）----
    for split_name, pattern in args.force_split_glob:
        pat = pattern.lower()
        matched = [k for k in img_map
                   if fnmatch.fnmatch(img_map[k].name.lower(), pat)
                   or fnmatch.fnmatch(k, pat)]
        if not matched:
            print(f'[警告] --force-split-glob {split_name}: 模式 {pattern!r} '
                  f'未匹配到任何图像', file=sys.stderr)
        for key in matched:
            prev = force_of_stem.get(key)
            if prev and prev != split_name:
                sys.exit(f'[错误] 同一图像被多种强制方式指定到不同子集: {key}'
                         f'（{prev} / {split_name}）')
            force_of_stem[key] = split_name

    # ---- 数据集构成：先打印全局总量与每个类别的图像数/框数（含占总数的百分比）----
    n_total = len(records)
    box_total = sum(len(r['lines']) for r in records)
    print(f'[数据] 共 {n_total} 张图 / {box_total} 个框，各类别数量'
          f'（一图含多类会重复计入，百分比 = 占图像/框总数）:')
    print(f'{"id":>4}  ' + _dw_ljust('类别', 16)
          + _dw_rjust('图像数(占比)', 16) + _dw_rjust('框数(占框总数)', 16))
    for cid, cname in enumerate(args.classes):
        n_img = sum(1 for r in records if cid in r['classes'])
        n_box = sum(r['per_class'].get(cid, 0) for r in records)
        print(f'{cid:>4}  ' + _dw_ljust(cname, 16)
              + _pct(n_img, n_total).rjust(16) + _pct(n_box, box_total).rjust(16))
    n_bg = sum(1 for r in records if r['is_bg'])
    if n_bg:
        print(f'{"bg":>4}  ' + _dw_ljust('背景图(空标签)', 16)
              + _pct(n_bg, n_total).rjust(16) + '-'.rjust(16))

    # ---- 分组（防泄漏核心：组 = 切分原子单位）----
    if args.no_dedup:
        groups = [[i] for i in range(len(records))]
        merged = capped = 0
    else:
        print(f'[分组] pHash 两两比对（阈值 {args.phash_threshold}'
              f'{f"，组上限 {args.max_group_size}" if args.max_group_size else ""}）...')
        groups, merged, capped = build_scene_groups(
            len(records), [r['hash'] for r in records], args.phash_threshold,
            args.max_group_size)

    # ---- 强制子集（--force-split / --force-split-glob）：组内原子，冲突即报错 ----
    stem_to_idx = {r['stem']: i for i, r in enumerate(records)}
    idx_force: Dict[int, str] = {}
    for k, v in force_of_stem.items():
        i = stem_to_idx.get(k)
        if i is None:
            print(f'[警告] 强制清单中的图像不可用（解码失败等），已忽略: {k}', file=sys.stderr)
        else:
            idx_force[i] = v
    group_force: Dict[int, str] = {}
    for gi, g in enumerate(groups):
        sset = sorted({idx_force[i] for i in g if i in idx_force})
        if len(sset) > 1:
            members = ', '.join(records[i]['path'].name for i in g)
            sys.exit(f'[错误] 同一组图像被强制到多个子集（组内原子，不可拆分）:\n'
                     f'       组成员: {members}\n'
                     f'       冲突子集: {" / ".join(sset)}\n'
                     f'       请统一指定，或调整 --phash-threshold / --no-dedup 改变分组。')
        if sset:
            group_force[gi] = sset[0]

    print('[切分] 稀有类优先 + 框数桶分层 ...')
    rng = random.Random(args.seed)
    split_of = assign_splits(records, groups, args.weights, rng, args.classes,
                             group_force)

    # test 未占占比但被强制指定时，仍生成 test 清单
    splits_active = ['train', 'val']
    if args.weights[2] > 0 or any(v == 'test' for v in group_force.values()):
        splits_active.append('test')
    splits_active = tuple(splits_active)
    print('[写出] labels / 清单 / data.yaml ...')
    write_outputs(out, labels_dir, records, split_of, args.classes, splits_active)
    copy_root: Optional[Path] = None
    if args.copy_out:
        copy_root = Path(args.copy_out).resolve()
        print(f'[拷贝] 生成自包含训练目录 -> {copy_root} ...')
        write_copyout_dataset(copy_root, records, split_of, args.classes,
                              splits_active, args.overwrite)
    print_report(args, records, split_of, groups, merged, stats, unknown,
                 parse_fail, probe_fail, splits_active, labels_dir,
                 group_force, capped, copy_root)


if __name__ == '__main__':
    main()
