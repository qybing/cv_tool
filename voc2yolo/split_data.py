r"""split_data.py — VOC 数据集切分（两步管线的第 1 步：先运行本脚本，再运行 voc_label.py）。

=== 使用流程（两个脚本的先后顺序）===
  第 1 步（本脚本）split_data.py：
    输入   <base>/Annotations/*.xml + <base>/JPEGImages/（或 images/）
    输出   <base>/ImageSets/Main/{train,val,test}.txt   切分清单（文件主干）
           <base>/label.txt                             类别名表（排序去重）
    做的事 自动发现类别 -> pHash 相似帧分组防泄漏 -> 稀有类优先分层切分

  第 2 步 voc_label.py：
    输入   第 1 步的 ImageSets/Main 清单 + Annotations + 图像
    输出   <base>/labels/*.txt（YOLO 标签）+ <base>/{train,val,test}.txt（绝对路径清单）
           + <base>/data.yaml（训练配置）
    做的事 VOC 像素坐标 -> YOLO 归一化坐标转换，并按第 1 步的类别表编号

  第 3 步 训练：
    yolo detect train data=<base>/data.yaml model=yolo26n.pt epochs=100 imgsz=640

运行前后目录结构对比：
  运行前:                    运行 split_data.py 后新增:
  VOC/                       VOC/
  ├─ Annotations/*.xml       ├─ Annotations/*.xml        （不变）
  └─ JPEGImages/*.jpg        ├─ JPEGImages/*.jpg         （不变）
                             ├─ ImageSets/Main/
                             │   ├─ train.txt  ├─ val.txt  ├─ test.txt
                             └─ label.txt
  （再运行 voc_label.py 后才会出现 labels/、images/、data.yaml、清单）

=== 参数逐项详解 ===
  --base / 顶部变量 DEFAULT_BASE          【唯一必填】
      VOC 根目录。脚本从这里找 Annotations/ 与图像目录。
      日常把它填进脚本顶部 DEFAULT_BASE 后，即可零参数运行。

  --images / DEFAULT_IMAGES（可选）
      图像目录。默认自动按 <base>/images -> <base>/JPEGImages 顺序查找，
      只有你的目录名不规范时才需要指定。防泄漏分组依赖图像，缺图像时
      自动退化为纯随机分层切分（会打印提示）。

  --anno / DEFAULT_ANNO（可选）
      VOC 标注目录。默认 <base>/Annotations。

  --ratios / DEFAULT_RATIOS（默认 0.9 0.1 0.0）
      train/val/test 占比。给 2 个数 = 没有 test；test 给 0 同效。
      不必严格加和为 1（自动归一化）。数据少时 test 可以一直留 0，
      平时只看 val 指标即可。

  --seed / DEFAULT_SEED（默认 666）
      随机种子。同种子 + 同数据 = 切分结果完全一致；对比实验期间不要改，
      否则 val 变了，前后指标不可比。

  --max-group-size / DEFAULT_MAX_GROUP_SIZE（默认 0 = 不限制）
      一个"组"最多几张。组 = pHash 判定的相似图像簇，是切分原子单位，
      组内图像永远进同一子集（防泄漏）。
      【不设的后果】30 张相似帧会绑成一个巨组，整批 all-or-nothing，
      实测可能整批进 val，模型一张都学不到。
      【建议值】批次张数 ÷ 10（30 张→3，40 张→4，长视频→100~200）。
      它只决定"切多细"，train/val 比例仍由 --ratios 决定。

  --phash-threshold / DEFAULT_PHASH_THRESHOLD（默认 8，范围 0~63）
      "多像才算同组"的汉明距离阈值。报告发现不同场景被误并 → 调小
      (5~6)；同一场景没并上 → 调大(10~12)。

  --force-split-glob <SPLIT> <通配符> / DEFAULT_FORCE_SPLIT_GLOB   （可重复传）
      文件名匹配通配符的图像【整组】强制进指定子集，免清单文件。
      如 --force-split-glob train "vid0912_*"。
      典型场景：误报/漏检批次（视频帧有固定前缀）主体锁进 train。
      组内原子：强制组里任意一张，整组跟随；同组被指向不同子集会报错。

  --force-split <SPLIT> <清单文件> / DEFAULT_FORCE_SPLIT            （可重复传）
      同上，但用清单文件指定：每行一个文件名（带不带扩展名均可，
      支持 # 注释）。可与 glob 叠加，组合出"主体锁 train + 抽样锁 val"。

  --no-dedup / DEFAULT_NO_DEDUP（默认 False）
      跳过防泄漏分组。视频帧/连拍数据【不要开】；只有真正独立照片
      （不同场景不同时间）才建议开，可省去全量算哈希的时间。

  --overwrite / DEFAULT_OVERWRITE（默认 False）
      重跑开关。ImageSets 清单或 label.txt 已存在时不带它会被拦截；
      改了标注或类别后重跑必带（或在脚本顶部设为 True 常开）。

  --sets-dir（可选）
      切分清单输出目录，默认 <base>/ImageSets/Main（VOC 惯例位置，
      voc_label.py 也默认从这里读，一般不要改）。

=== 特性 ===
  - 类别自动从 XML 发现，无需手工维护类别表；
  - pHash 相似帧分组防泄漏（连拍/视频抽帧的相邻帧整组进同一子集）；
  - 稀有类优先分层 + 框数桶分层，保证稀有类在 val/test 有样本；
  - 固定种子可复现、覆盖保护、逐项异常统计与报告。

=== 用法示例 ===
  # 日常用法：在脚本顶部 DEFAULT_BASE 填好路径后，直接零参数运行
  python split_data.py

  python split_data.py --base D:\data\VOC
  python split_data.py --base D:\data\VOC --ratios 0.8 0.1 0.1 --max-group-size 5
  python split_data.py --base D:\data\VOC --no-dedup            # 独立照片提速

  # 典型误报批次迭代（30 张级相似帧，主体锁 train）：
  python split_data.py --overwrite --max-group-size 5 --force-split-glob train "vid0912_*"

下一步：python voc_label.py --base D:\data\VOC   （转 YOLO 格式）
依赖：pillow（防泄漏分组）、numpy（仅分组）；--no-dedup 时纯标准库。
本脚本自包含（内置切分引擎），可单独拷贝到任何机器使用。
"""
import argparse
import fnmatch
import random
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ============================================================================
# 可调参数集中区
# ============================================================================

# ---- 默认配置：不带参数直接运行时生效，命令行传参可覆盖 ----
DEFAULT_BASE = r''                 # VOC 根目录（含 Annotations/ 与 JPEGImages/ 或 images/）
DEFAULT_IMAGES = r''               # 图像目录（留空 = 依次找 <base>/images、<base>/JPEGImages）
DEFAULT_ANNO = r''                 # 标注目录（留空 = <base>/Annotations）
DEFAULT_RATIOS = (0.9, 0.1, 0.0)   # train/val/test 占比（test 为 0 = 不生成 test 集）
DEFAULT_SEED = 666                 # 随机种子（同种子同数据 = 切分完全可复现）
DEFAULT_MAX_GROUP_SIZE = 0         # 组最大张数（30 张级误报批次建议 3~5；0 = 不限制）
DEFAULT_PHASH_THRESHOLD = 8        # 相似判定阈值：汉明距离 <= 它归为同组（0~63）
DEFAULT_NO_DEDUP = False           # True = 跳过防泄漏分组（仅独立照片数据用）
DEFAULT_OVERWRITE = False          # True = 已有切分结果时直接覆盖重建
DEFAULT_FORCE_SPLIT_GLOB: tuple = ()   # 按通配符强制子集，如 (('train', r'vid0912_*'),)
DEFAULT_FORCE_SPLIT: tuple = ()        # 按清单强制子集，如 (('train', r'D:\stems.txt'),)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
SPLIT_NAMES = ('train', 'val', 'test')            # 子集名（VOC 惯例，勿改）

PHASH_IMG_SIZE = 32                # 计算哈希前灰度图缩放尺寸
PHASH_LOW_FREQ = 8                 # DCT 低频 N x N 系数 -> 64 位哈希
HASH_RETRY_TIMES = 3               # 图像读取重试次数（杀软/索引器瞬时占用）
HASH_RETRY_SLEEP = 0.2             # 重试等待基数（秒），第 n 次等 n * 该值
COMPARE_BUDGET_CELLS = 4_000_000   # 距离矩阵单块元素预算（约 32MB）
COMPARE_CHUNK_CAP = 4096           # 单块最大行数
COMPARE_PROGRESS_EVERY = 8000      # 比对进度打印阈值
BUCKET_MIN_GROUPS = 15             # 轮次池组数达到该值才按框数桶分层
SMALL_POOL_IMAGES = 8              # 池内图像数不超过该值时按组数分配
LABEL_DECIMALS = 6                 # YOLO 标签坐标小数位数（内部解析用，保持一致性）

PHASH_MAX_DISTANCE = PHASH_LOW_FREQ ** 2 - 1  # 汉明距离合法上限（63）


class _SizeMissing(ValueError):
    """XML 缺少有效尺寸且未提供兜底尺寸。"""


# ------------------------------------------------------------------ 表格对齐
def dw_width(text: str) -> int:
    """终端显示宽度：中日韩字符按 2 列，其余 1 列。"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def dw_ljust(text: str, width: int) -> str:
    """按显示宽度左对齐。"""
    return text + ' ' * max(0, width - dw_width(text))


def dw_rjust(text: str, width: int) -> str:
    """按显示宽度右对齐。"""
    return ' ' * max(0, width - dw_width(text)) + text


def pct(n: int, denom: int) -> str:
    """格式化 "n (x.x%)"；分母为 0 时百分比记 0.0%。"""
    return f'{n} ({n / denom:.1%})' if denom else f'{n} (0.0%)'


# ------------------------------------------------------------------ 文件扫描
def scan_files(directory: Path, exts: Set[str]) -> Dict[str, Path]:
    """扫描目录（不递归），返回 {文件主干(小写): 路径}；主干重名保留排序靠前者。"""
    mapping: Dict[str, Path] = {}
    for f in sorted(Path(directory).iterdir()):
        if f.is_file() and f.suffix.lower() in exts:
            key = f.stem.lower()
            if key in mapping:
                print(f'[警告] 文件主干重名，忽略后者: {mapping[key].name} / {f.name}',
                      file=sys.stderr)
                continue
            mapping[key] = f
    return mapping


# ------------------------------------------------------------------ XML 解析
def parse_xml_root(xml_path: Path):
    """解析 XML 根节点，带编码回退（声明编码 -> utf-8-sig -> gbk -> latin-1）。"""
    try:
        return ET.parse(str(xml_path)).getroot()
    except ET.ParseError:
        raw = Path(xml_path).read_bytes()
        for enc in ('utf-8-sig', 'gbk', 'latin-1'):
            try:
                return ET.fromstring(raw.decode(enc))
            except (UnicodeDecodeError, ET.ParseError):
                continue
        raise ValueError(f'XML 解析失败（已尝试 utf-8/gbk/latin-1）: '
                         f'{Path(xml_path).name}') from None


def parse_voc_xml(xml_path: Path, class_ids: Dict[str, int],
                  fallback_size: Optional[Tuple[int, int]],
                  skip_difficult: bool, unknown: Counter):
    """解析单个 VOC XML。

    Returns:
        (YOLO 标签行列表, 类别 id 集合, 每类框数 Counter, 本文件统计)。
        统计键: difficult / unknown / badbox / clipped / size_fallback。
    Raises:
        _SizeMissing: 尺寸缺失且无兜底；ValueError: XML 无法解析。
    """
    root = parse_xml_root(xml_path)
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
            w, h = fallback_size              # 用图像实际尺寸兜底，不丢标注
            st['size_fallback'] = 1
        else:
            raise _SizeMissing('XML 缺少有效尺寸，且未提供兜底尺寸')

    lines: List[str] = []
    classes: Set[int] = set()
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
        # 越界坐标裁剪到图像范围内，退化框（宽/高 <= 0）跳过
        cx0, cy0 = min(max(x0, 0.0), float(w)), min(max(y0, 0.0), float(h))
        cx1, cy1 = min(max(x1, 0.0), float(w)), min(max(y1, 0.0), float(h))
        if not (cx1 > cx0 and cy1 > cy0):
            st['badbox'] += 1
            continue
        if (cx0, cy0, cx1, cy1) != (x0, y0, x1, y1):
            st['clipped'] += 1
        cid = class_ids[name]
        xc = (cx0 + cx1) / 2.0 / w
        yc = (cy0 + cy1) / 2.0 / h
        bw = (cx1 - cx0) / w
        bh = (cy1 - cy0) / h
        lines.append(f'{cid} {xc:.{LABEL_DECIMALS}f} {yc:.{LABEL_DECIMALS}f} '
                     f'{bw:.{LABEL_DECIMALS}f} {bh:.{LABEL_DECIMALS}f}')
        classes.add(cid)
        per_class[cid] += 1
    return lines, classes, per_class, st


def discover_classes(xml_map: Dict[str, Path]) -> List[str]:
    """轻扫描全部 XML，收集并排序所有类别名（供切分阶段自动分层）。"""
    names: Set[str] = set()
    for p in xml_map.values():
        try:
            root = parse_xml_root(p)
        except ValueError as e:
            print(f'[警告] {e}', file=sys.stderr)
            continue
        for obj in root.findall('object'):
            n = (obj.findtext('name') or '').strip()
            if n:
                names.add(n)
    return sorted(names)


# ------------------------------------------------------------------ 图像探测
def _phash_from_gray(a) -> int:
    """32x32 灰度矩阵 -> 64 位 pHash（DCT 低频按中值二值化，直流位恒 0）。"""
    import numpy as np
    n = PHASH_IMG_SIZE
    idx = np.arange(n)
    dct = np.sqrt(2.0 / n) * np.cos(np.pi * (2 * idx[None, :] + 1) * idx[:, None] / (2 * n))
    dct[0, :] = np.sqrt(1.0 / n)
    freq = dct @ a @ dct.T
    low = freq[:PHASH_LOW_FREQ, :PHASH_LOW_FREQ].flatten()
    bits = low > np.median(low[1:])       # 中值不含直流分量
    bits[0] = False
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return int(v)


def probe_image(img_path: Path, need_hash: bool):
    """打开图像一次，返回 (pHash 或 None, (宽, 高) 或 None, 失败原因 或 None)。

    need_hash=False 时只读文件头取尺寸（快）；OSError 短暂重试，
    其他异常（如分辨率炸弹）不重试；失败时保留已拿到的部分。
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
                wh = im.size                          # 惰性读取，不解码像素
                if need_hash:
                    gray = im.convert('L').resize(
                        (PHASH_IMG_SIZE, PHASH_IMG_SIZE), Image.LANCZOS)
                    phash_val = _phash_from_gray(np.asarray(gray, dtype=np.float64))
            return phash_val, wh, None
        except OSError as e:
            err = f'{type(e).__name__}: {e}'
            time.sleep(HASH_RETRY_SLEEP * (attempt + 1))
        except Exception as e:                        # 解码库异常类型不可枚举
            return None, wh, f'{type(e).__name__}: {e}'
    return phash_val, wh, err


# ------------------------------------------------------------------ 场景分组
class _UnionFind:
    """轻量并查集（路径压缩 + 可选组大小上限）。"""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int, max_size: int = 0) -> bool:
        """合并两个集合；同集合或超 max_size 上限时返回 False。"""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if max_size and self.size[ra] + self.size[rb] > max_size:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def build_scene_groups(n_records: int, hashes: List[Optional[int]], threshold: int,
                       max_group_size: int = 0):
    """pHash 汉明距离 <= threshold 连边，并查集求连通分量。

    哈希为 None（解码失败）的记录自成一组。返回 (组列表, 合并边数,
    因大小上限被拒绝的合并对数)；组列表按首成员索引排序，保证确定性。
    """
    import numpy as np
    ok = [i for i, h in enumerate(hashes) if h is not None]
    uf = _UnionFind(n_records)
    merged = capped = 0
    m = len(ok)
    if m >= 2:
        hv = np.array([hashes[i] for i in ok], dtype=np.uint64)
        lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
        chunk = max(1, min(COMPARE_CHUNK_CAP, COMPARE_BUDGET_CELLS // m))
        for start in range(0, m, chunk):
            block = hv[start:start + chunk]
            xor = block[:, None] ^ hv[None, :]
            codes = xor.view(np.uint8).reshape(len(block), -1, 8)
            dist = lut[codes].sum(axis=2)
            for li, lj in np.argwhere(dist <= threshold):
                i, j = ok[start + int(li)], ok[int(lj)]
                if i < j and uf.find(i) != uf.find(j):
                    if uf.union(i, j, max_group_size):
                        merged += 1
                    else:
                        capped += 1
            if m > COMPARE_PROGRESS_EVERY:
                print(f'  比对进度 {min(start + len(block), m)}/{m}')
    members = defaultdict(list)
    for i in range(n_records):
        members[uf.find(i)].append(i)
    groups = sorted(members.values(), key=lambda g: g[0])
    return groups, merged, capped


# ------------------------------------------------------------------ 记录收集
def collect_records(img_map: Dict[str, Path], xml_map: Dict[str, Path],
                    class_ids: Dict[str, int], skip_difficult: bool,
                    dedup: bool, include_background: bool):
    """匹配图像与 XML 并解析为记录列表。

    Returns:
        (records, stats, unknown, parse_fail, probe_fail)。
        record 字段: stem / path / hash / lines / classes / per_class / is_bg。
    """
    records: List[dict] = []
    stats: Counter = Counter()
    unknown: Counter = Counter()
    parse_fail: List[str] = []
    probe_fail: List[str] = []
    size_cache: Dict[str, Tuple[int, int]] = {}

    hashes: Dict[str, Optional[int]] = {}
    if dedup:
        n_total = len(img_map)
        print(f'[去重] 计算 {n_total} 张图像的 pHash ...')
        for k, stem in enumerate(sorted(img_map), 1):
            h, wh, err = probe_image(img_map[stem], need_hash=True)
            if wh is None:
                probe_fail.append(f'{img_map[stem].name}（{err or "无法读取"}，已排除）')
                continue
            if h is None:
                probe_fail.append(f'{img_map[stem].name}（{err or "哈希失败"}，按独立组处理）')
            hashes[stem] = h
            size_cache[stem] = wh
            if n_total > COMPARE_PROGRESS_EVERY and k % 1000 == 0:
                print(f'  已计算 {k}/{n_total}')

    for stem in sorted(img_map):
        if dedup and stem not in size_cache:
            continue                          # 探测阶段已判死（无法解码）
        img_path = img_map[stem]
        xml_path = xml_map.get(stem)
        lines: List[str] = []
        classes: Set[int] = set()
        per_class: Counter = Counter()

        if xml_path is None:
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
                stats['empty_xml_bg'] += 1    # XML 存在但无有效目标 -> 背景图

        records.append({
            'stem': stem, 'path': img_path, 'hash': hashes.get(stem),
            'lines': lines, 'classes': classes, 'per_class': per_class,
            'is_bg': not classes,
        })
    return records, stats, unknown, parse_fail, probe_fail


# ------------------------------------------------------------------ 分层切分
def alloc_counts(n: int, weights) -> List[int]:
    """最大余数法按 weights 分配 n 个样本，并保证各桶尽量非空（train>val>test 优先）。"""
    k = len(weights)
    quotas = [n * w for w in weights]
    counts = [int(q) for q in quotas]
    rem = n - sum(counts)
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
    """分组 + 稀有类优先 + 框数桶分层的切分，返回与 records 对齐的子集名列表。"""
    r_train, r_val, r_test = weights

    g_classes, g_boxes, g_size = [], [], []
    for g in groups:
        cs: Set[int] = set()
        boxes = 0
        for i in g:
            cs |= records[i]['classes']
            boxes += sum(records[i]['per_class'].values())
        g_classes.append(cs)
        g_boxes.append(boxes)
        g_size.append(len(g))

    cls_groups = defaultdict(list)
    for gi, cs in enumerate(g_classes):
        for c in cs:
            cls_groups[c].append(gi)
    cls_n = {c: len(v) for c, v in cls_groups.items()}

    def round_of(cs):
        """组 -> 轮次类别：取组内最稀有类（平局按类别 id，保证确定性）。"""
        if not cs:
            return None
        return min(cs, key=lambda c: (cls_n[c], c))

    g_round = [round_of(cs) for cs in g_classes]
    order = sorted(cls_n, key=lambda c: (cls_n[c], c)) + [None]
    split_of_group: Dict[int, str] = dict(forced or {})

    def assign_pool(pool: List[int]) -> None:
        """把一个池（已洗牌）按策略装填到三个子集。"""
        total_imgs = sum(g_size[gi] for gi in pool)
        if total_imgs <= SMALL_POOL_IMAGES:
            counts = alloc_counts(len(pool), (r_train, r_val, r_test))
            seq = (['train'] * counts[0] + ['val'] * counts[1]
                   + ['test'] * counts[2])
            for gi, place in zip(pool, seq):
                split_of_group[gi] = place
            return
        # 大池：val/test 设图像配额上限（带容差），装不下的默认进 train
        caps = {'val': round(total_imgs * r_val),
                'test': round(total_imgs * r_test) if r_test > 0 else 0}
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
        return any(split_of_group.get(gi) == target for gi in cls_groups[c])

    for r in order:
        pool = [gi for gi in range(len(groups))
                if g_round[gi] == r and gi not in split_of_group]
        if not pool:
            continue
        rng.shuffle(pool)
        pool_imgs = sum(g_size[gi] for gi in pool)
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
        # 兜底：本类 val/test 仍无组时，从 train 挪最小的组补上
        if r is not None:
            targets = [('val', 2)] + ([('test', 3)] if r_test > 0 else [])
            for target, min_groups in targets:
                if cls_n[r] >= min_groups and not covered(r, target):
                    cands = [gi for gi in cls_groups[r]
                             if gi not in (forced or {})
                             and split_of_group.get(gi) == 'train']
                    if cands:
                        gi = min(cands, key=lambda g: (g_size[g], g))
                        split_of_group[gi] = target

    # 全局兜底：极端情况下 val/test 为空时从 train 挪最小的组
    for target, min_n in [('val', 2)] + ([('test', 3)] if r_test > 0 else []):
        if len(records) >= min_n and not any(v == target for v in split_of_group.values()):
            cands = [gi for gi, v in split_of_group.items()
                     if gi not in (forced or {}) and v == 'train']
            if cands:
                gi = min(cands, key=lambda g: (g_size[g], g))
                split_of_group[gi] = target

    split_of: List[str] = [''] * len(records)
    for gi, g in enumerate(groups):
        for i in g:
            split_of[i] = split_of_group[gi]
    return split_of


# ------------------------------------------------------------------ 命令行
def parse_args() -> argparse.Namespace:
    """解析并校验命令行参数。"""
    p = argparse.ArgumentParser(
        prog='split_data.py',
        description='VOC 数据集切分（ImageSets/Main + 类别表），'
                    '带 pHash 防泄漏与稀有类优先分层',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--base', type=Path, default=(DEFAULT_BASE or None),
                   help='VOC 根目录（默认取脚本顶部 DEFAULT_BASE）')
    p.add_argument('--images', type=Path, default=(DEFAULT_IMAGES or None),
                   help='图像目录（默认依次找 <base>/images、<base>/JPEGImages）')
    p.add_argument('--anno', type=Path, default=(DEFAULT_ANNO or None),
                   help='标注目录（默认 <base>/Annotations）')
    p.add_argument('--ratios', nargs='+', type=float, default=list(DEFAULT_RATIOS),
                   metavar='R', help='train/val/test 占比（2 或 3 个数，test 可为 0）')
    p.add_argument('--sets-dir', type=Path, default=None,
                   help='切分清单输出目录（默认 <base>/ImageSets/Main）')
    p.add_argument('--seed', type=int, default=DEFAULT_SEED,
                   help='随机种子，保证切分可复现')
    p.add_argument('--no-dedup', action='store_true', default=DEFAULT_NO_DEDUP,
                   help='跳过 pHash 防泄漏分组（默认取顶部 DEFAULT_NO_DEDUP；'
                        '独立照片数据用，连拍/抽帧不建议开）')
    p.add_argument('--phash-threshold', type=int, default=DEFAULT_PHASH_THRESHOLD,
                   metavar='D',
                   help=f'近似重复判定阈值：汉明距离 <= D 归为同组'
                        f'（0~{PHASH_MAX_DISTANCE}，越大合并越激进）')
    p.add_argument('--max-group-size', type=int, default=DEFAULT_MAX_GROUP_SIZE,
                   metavar='K',
                   help='组最大张数：连通分量超过 K 张时按相似序拆分'
                        '（30 张级批次建议 3~5，长视频建议 100~200；0 = 不限制）')
    p.add_argument('--overwrite', action='store_true', default=DEFAULT_OVERWRITE,
                   help='ImageSets 清单已存在时覆盖重建（默认取顶部 DEFAULT_OVERWRITE）')
    p.add_argument('--force-split-glob', action='append', nargs=2,
                   metavar=('SPLIT', 'PATTERN'),
                   default=list(DEFAULT_FORCE_SPLIT_GLOB),
                   help='按文件名通配符【整组】强制进指定子集（免清单），'
                        '如 --force-split-glob train "vid0912_*"；可重复传')
    p.add_argument('--force-split', action='append', nargs=2,
                   metavar=('SPLIT', 'FILE'), default=list(DEFAULT_FORCE_SPLIT),
                   help='按清单文件【整组】强制进指定子集（每行一个文件名，'
                        '支持 # 注释）；可重复传')
    args = p.parse_args()

    if args.base is None:
        p.error('未指定 VOC 根目录：请在脚本顶部 DEFAULT_BASE 填写路径，'
                '或用 --base 传入')

    args.base = args.base.resolve()
    args.anno = (args.anno or args.base / 'Annotations').resolve()
    if not args.anno.is_dir():
        p.error(f'标注目录不存在: {args.anno}')
    if args.images is not None:
        args.images = args.images.resolve()
        if not args.images.is_dir():
            p.error(f'图像目录不存在: {args.images}')

    r = list(args.ratios)
    if len(r) not in (2, 3):
        p.error('--ratios 需要提供 2 个（train val）或 3 个数（train val test）')
    if len(r) == 2:
        r.append(0.0)
    if any(x < 0 for x in r):
        p.error('占比不能为负数')
    total = sum(r)
    if total <= 0:
        p.error('占比之和必须大于 0')
    if abs(total - 1.0) > 0.02:
        print(f'[提示] 占比之和为 {total:.3f}，已自动归一化', file=sys.stderr)
    args.weights = [x / total for x in r]
    if args.weights[0] <= 0 or args.weights[1] <= 0:
        p.error('train 与 val 占比必须大于 0（test 允许为 0）')
    if not (0 <= args.phash_threshold <= PHASH_MAX_DISTANCE):
        p.error(f'--phash-threshold 取值范围 0~{PHASH_MAX_DISTANCE}')
    if args.max_group_size < 0:
        p.error('--max-group-size 不能为负（0 表示不限制）')
    for split_name, _pattern in args.force_split_glob:
        if split_name not in SPLIT_NAMES:
            p.error(f'--force-split-glob 子集名必须是 {" / ".join(SPLIT_NAMES)}，'
                    f'收到: {split_name}')
    for split_name, list_path in args.force_split:
        if split_name not in SPLIT_NAMES:
            p.error(f'--force-split 子集名必须是 {" / ".join(SPLIT_NAMES)}，'
                    f'收到: {split_name}')
        if not Path(list_path).is_file():
            p.error(f'--force-split 清单文件不存在: {list_path}')
    return args


def print_report(args, records, split_of, groups, stats,
                 parse_fail, probe_fail, classes, sets_dir,
                 no_img_names=None, forced=None) -> None:
    """打印切分统计表与异常提示。"""
    total = len(records)
    counts = {s: sum(1 for v in split_of if v == s) for s in SPLIT_NAMES}
    print('=' * 66)
    print('数据集切分完成')
    print('=' * 66)
    print(f'切分清单 : {sets_dir}')
    print(f'类别表   : {args.base / "label.txt"}（{len(classes)} 个类别，'
          f'供 voc_label.py 使用）')
    print(f'随机种子 : {args.seed}（对比实验请保持一致）')
    print(f'图像总数 : {total} 张')
    print('切分数量 : ' + '  '.join(
        pct(counts[s], total).rjust(12) for s in SPLIT_NAMES).strip())
    n_multi = sum(1 for g in groups if len(g) > 1)
    if args.no_dedup:
        print('去重分组 : 已跳过（--no-dedup）')
    else:
        print(f'去重分组 : {len(groups)} 组（多图组 {n_multi} 个，'
              f'最大组 {max(len(g) for g in groups)} 张）——组内图像固定进同一子集')
    if forced:
        parts = []
        for s in SPLIT_NAMES:
            gs = [gi for gi, sp in forced.items() if sp == s]
            if gs:
                parts.append(f'{s}={len(gs)}组/{sum(len(groups[gi]) for gi in gs)}图')
        print('强制指定 : ' + ('  '.join(parts) if parts else '-')
              + '（--force-split，组内原子生效）')

    print('-' * 66)
    print('各类别图像分布（一图含多类重复计入，百分比 = 占图像总数）:')
    w = 13
    print(f'{"id":>3}  ' + dw_ljust('类别', 16)
          + ''.join(dw_rjust(s, w) for s in SPLIT_NAMES)
          + dw_rjust('合计', w) + '   备注')
    for cid, cname in enumerate(classes):
        imgs = {s: sum(1 for r, v in zip(records, split_of)
                       if v == s and cid in r['classes']) for s in SPLIT_NAMES}
        tot = sum(imgs.values())
        notes = []
        if imgs['val'] == 0:
            notes.append('val 无样本!')
        if imgs['test'] == 0 and args.weights[2] > 0:
            notes.append('test 无样本')
        if tot < 10:
            notes.append('样本过少(<10)')
        print(f'{cid:>3}  ' + dw_ljust(cname, 16)
              + ''.join(dw_rjust(pct(imgs[s], total), w) for s in SPLIT_NAMES)
              + dw_rjust(pct(tot, total), w)
              + '   ' + ('[' + '; '.join(notes) + ']' if notes else ''))

    # 背景图行：XML 存在但无有效目标（空标签），参与训练可抑制误检
    n_bg = sum(1 for r in records if r['is_bg'])
    if n_bg:
        bg = {s: sum(1 for r, v in zip(records, split_of)
                     if v == s and r['is_bg']) for s in SPLIT_NAMES}
        print(f'{"bg":>3}  ' + dw_ljust('背景图(空标签)', 16)
              + ''.join(dw_rjust(pct(bg[s], total), w) for s in SPLIT_NAMES)
              + dw_rjust(pct(n_bg, total), w))

    print('-' * 66)
    msgs = []
    if stats['xml_no_img']:
        names = (no_img_names or [])[:5]
        show = ', '.join(names) + ('...' if stats['xml_no_img'] > 5 else '')
        msgs.append(f"{stats['xml_no_img']} 个 XML 无对应图像，已忽略: {show}")
    if stats['no_xml_excluded']:
        msgs.append(f"{stats['no_xml_excluded']} 张图像无标注文件，未参与切分")
    if stats['empty_xml_bg']:
        msgs.append(f"{stats['empty_xml_bg']} 个 XML 无有效目标，按背景图参与切分")
    if stats['size_fallback']:
        msgs.append(f"{stats['size_fallback']} 个 XML 尺寸缺失/非法，"
                    f"已用图像实际尺寸兜底")
    if parse_fail:
        show = '; '.join(parse_fail[:5]) + ('...' if len(parse_fail) > 5 else '')
        msgs.append(f'{len(parse_fail)} 个 XML 解析失败被排除: {show}')
    if probe_fail:
        show = '; '.join(probe_fail[:5]) + ('...' if len(probe_fail) > 5 else '')
        msgs.append(f'{len(probe_fail)} 张图像读取/解码异常: {show}')
    if stats['badbox']:
        msgs.append(f"跳过退化/无效目标框 {stats['badbox']} 个")
    if stats['clipped']:
        msgs.append(f"裁剪越界目标框 {stats['clipped']} 个")
    if msgs:
        for m in msgs:
            print(f'[提示] {m}')
    else:
        print('[提示] 未发现异常')
    print('=' * 66)
    print('下一步：python voc_label.py --base '
          f'"{args.base}"   （转 YOLO 格式）')


def main() -> None:
    args = parse_args()
    anno_map = scan_files(args.anno, {'.xml'})
    if not anno_map:
        sys.exit(f'[错误] 标注目录中没有 XML: {args.anno}')

    # 图像目录解析：显式指定 > <base>/images > <base>/JPEGImages
    img_dir: Optional[Path] = args.images
    if img_dir is None:
        for cand in (args.base / 'images', args.base / 'JPEGImages'):
            if cand.is_dir():
                img_dir = cand
                break
    img_map = scan_files(img_dir, IMG_EXTS) if img_dir else {}
    if not img_map:
        print('[提示] 未找到图像目录，跳过防泄漏分组（仅按 XML 随机分层切分）',
              file=sys.stderr)

    sets_dir = args.sets_dir or (args.base / 'ImageSets' / 'Main')
    existing = [sets_dir / f'{s}.txt' for s in SPLIT_NAMES
                if (sets_dir / f'{s}.txt').is_file()]
    label_file = args.base / 'label.txt'
    if (existing or label_file.is_file()) and not args.overwrite:
        sys.exit(f'[错误] 切分结果已存在: {sets_dir} / {label_file}\n'
                 f'       确认覆盖重建请加 --overwrite 重新运行。')
    sets_dir.mkdir(parents=True, exist_ok=True)

    # 类别自动发现（排序去重，保证确定性），全量作为已知类别参与解析
    classes = discover_classes(anno_map)
    if not classes:
        sys.exit('[错误] 未在 XML 中发现任何类别名')
    class_ids = {c: i for i, c in enumerate(classes)}

    dedup = (not args.no_dedup) and bool(img_map)
    records, stats, unknown, parse_fail, probe_fail = collect_records(
        img_map, anno_map, class_ids, skip_difficult=False,
        dedup=dedup, include_background=False)
    no_img_names = sorted(set(anno_map) - set(img_map))
    stats['xml_no_img'] = len(no_img_names)
    if not records:
        sys.exit('[错误] 没有匹配到任何「图像 + 标注」数据，请检查文件名是否对应')

    # ---- 解析强制子集（--force-split-glob / --force-split）：主干 -> 子集 ----
    force_of_stem: Dict[str, str] = {}
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
    for split_name, list_path in args.force_split:
        with open(list_path, encoding='utf-8-sig') as fh:
            stems = [ln.strip() for ln in fh
                     if ln.strip() and not ln.strip().startswith('#')]
        miss = [s for s in stems if Path(s).stem.lower() not in img_map]
        if miss:
            show = ', '.join(miss[:5]) + ('...' if len(miss) > 5 else '')
            print(f'[警告] --force-split {split_name}: {len(miss)} 个文件名不在'
                  f'数据集中，已忽略: {show}', file=sys.stderr)
        for s in stems:
            key = Path(s).stem.lower()      # 兼容带/不带扩展名
            if key not in img_map:
                continue
            prev = force_of_stem.get(key)
            if prev and prev != split_name:
                sys.exit(f'[错误] 同一图像被多种强制方式指定到不同子集: {key}'
                         f'（{prev} / {split_name}）')
            force_of_stem[key] = split_name

    # 分组与切分
    if dedup:
        print(f'[分组] pHash 两两比对（阈值 {args.phash_threshold}'
              f'{f"，组上限 {args.max_group_size}" if args.max_group_size else ""}）...')
        groups, merged, capped = build_scene_groups(
            len(records), [r['hash'] for r in records],
            args.phash_threshold, args.max_group_size)
    else:
        groups, merged, capped = [[i] for i in range(len(records))], 0, 0

    # 强制子集升级到组级（组内原子：整组跟随；同组冲突直接报错）
    stem_to_idx = {r['stem']: i for i, r in enumerate(records)}
    idx_force: Dict[int, str] = {}
    for k, v in force_of_stem.items():
        i = stem_to_idx.get(k)
        if i is None:
            print(f'[警告] 强制清单中的图像不可用（解码失败等），已忽略: {k}',
                  file=sys.stderr)
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
                     f'       请统一指定，或调整 --phash-threshold / --max-group-size。')
        if sset:
            group_force[gi] = sset[0]

    print('[切分] 稀有类优先分层 ...')
    rng = random.Random(args.seed)
    split_of = assign_splits(records, groups, args.weights, rng, classes,
                             group_force)

    # 写 ImageSets/Main 清单（三个文件都写，未启用子集为空文件，保持 VOC 惯例）
    for s in SPLIT_NAMES:
        stems = sorted(r['stem'] for r, v in zip(records, split_of) if v == s)
        (sets_dir / f'{s}.txt').write_text(
            '\n'.join(stems) + ('\n' if stems else ''), encoding='utf-8')
    # 类别表：排序去重（确定性），供 voc_label.py 读取
    (args.base / 'label.txt').write_text(
        '\n'.join(classes) + '\n', encoding='utf-8')

    print_report(args, records, split_of, groups, stats,
                 parse_fail, probe_fail, classes, sets_dir, no_img_names,
                 group_force)


if __name__ == '__main__':
    main()
