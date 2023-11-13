# -*- coding: utf-8 -*-

"""
辅助工具函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import pandas
import numpy
import scipy.sparse
import sklearn.preprocessing
import sklearn.impute
import sklearn.feature_extraction.text
import matplotlib
import shapely
import cartopy


def make_dict(data, minfreq=None, sort=True):
    """
    根据方言数据构建词典.

    Parameters:
        data (array-like): 读音样本数据中的一列，空字符串代表缺失值
        minfreq (float or int): 出现频次不小于该值才计入词典
        sort (bool): 返回的词典按顺序排列

    Returns:
        dic (`numpy.ndarray`): data 中符号的词典
    """

    dic, counts = numpy.unique(data[data != ''], return_counts=True)

    if minfreq is not None and minfreq > 1:
        # 如果 minfreq 是实数，指定出现的最小比例
        dic = dic[counts >= (int(minfreq * len(data)) if isinstance(minfreq) \
            else minfreq)]

    if sort:
        dic.sort()

    return dic

def split_data(values, *data, test_size=0.1, return_mask=False, random_state=None):
    """
    根据数值及比例切分数据.

    Parameters:
        values (array-like): 待切分数据
        data (array-like): 额外的待切分数据
        test_size (float): 切分后测试集占样本的比例
        return_mask (bool): 为真时返回切分的结果掩码
        random_state (int or `numpy.random.RandomState`): 用于复现划分结果

    Returns:
        train_values, test_values, ...: 切分后的数据

    从 values 的取值中随机选择一批用作测试，使切分后的训练集不包含该值，测试集只包含该值，
    且测试集占比大致等于 test_size。为增加测试集的多样性，尽可能选择出现次数少的值用作测试集。
    """

    if random_state is None:
        random_state = numpy.random.mtrand._rand
    elif isinstance(random_state, int):
        random_state = numpy.random.RandomState(random_state)

    # 计算每个值在样本中出现的比例及反向索引
    _, inverse, counts = numpy.unique(
        values,
        return_inverse=True,
        return_counts=True
    )
    ratio = counts / len(values)
    # 根据比例及一定随机性排定值进入测试集的优先级
    prior = ratio + random_state.normal(scale=numpy.std(ratio), size=ratio.shape[0])
    # 根据优先级排序，根据在总样本中的占比截取需要的数量
    mask = numpy.empty(ratio.shape, dtype=numpy.bool8)
    idx = numpy.argsort(prior)
    mask[idx] = ratio[idx].cumsum() < test_size
    # 保证训练集、测试集均不为空
    mask[idx[0]] = True
    mask[idx[-1]] = False
    # 把切分结果根据反向索引映射回原始样本
    mask = mask[inverse]

    if return_mask:
        # 返回训练集、测试集的掩码
        return ~mask, mask

    else:
        # 返回切分结果
        results = [values[~mask], values[mask]]
        for d in data:
            results.extend(d[~mask], d[mask])

        return results

def encode(data, dtype=numpy.int32, missing_values='', unknown_value=-1):
    """
    把方言读音编码为整数.

    Parameters:
        data (`pandas.DataFrame`): M x N 字符串矩阵，每行为一个字，
            每列为一个方言点的声母/韵母/声调，空串代表空值

    Returns:
        codes (`numpy.ndarray`): M x N 整数矩阵，空值为 -1
        categories (list of `numpy.ndarray`): 长度为 N 的列表，每个元素是每一列的类别
    """

    encoder = sklearn.preprocessing.OrdinalEncoder(
        dtype=dtype,
        handle_unknown='use_encoded_value',
        unknown_value=unknown_value
    )
    encoder.fit(
        # 为了让编码器正常工作，先补全缺失特征
        sklearn.impute.SimpleImputer(
            missing_values=missing_values,
            strategy='most_frequent'
        ).fit_transform(data)
    )

    return encoder.transform(data), encoder.categories_

def vectorize(data, sep=' ', binary=False, dtype=numpy.int32, norm=None):
    """
    对一个方言读音的数组或包含多个方言读音的矩阵进行稀疏编码.

    原始数据以字为行，以方言点的声韵调为列，允许一格包含多个音，以指定分隔符分隔。

    Parameters:
        data (array): 长度为 M 的数组 或 M x N 矩阵，当为矩阵时，每列为一个方言点的声母/韵母/声调
        sep (str): 分隔多音字的多个音的分隔符
        binary (bool): 为真时，返回的编码为 0/1 编码，否则返回读音的计数
        norm (str): 是否对返回编码归一化：
            - None: 不归一化
            - 'l1': 返回的编码除以向量的1范数
            - 'l2': 返回的编码除以向量的2范数

    Returns:
        code (`scipy.sparse.csr_matrix`): 稀疏编码得到的稀疏矩阵，行数为 M，列数为所有列读音数之和
        limits (`numpy.ndarray`): 仅当 data 为矩阵时返回，表示编码边界的数组，
            长度为 N + 1，data[:, i] 的编码为 code[:, limits[i]:limits[i + 1]]
    """

    if isinstance(data, pandas.DataFrame) or isinstance(data, pandas.Series):
        data = data.values

    if data.ndim == 1:
        # 一维数组，直接编码返回
        code = sklearn.feature_extraction.text.CountVectorizer(
            lowercase=False,
            tokenizer=lambda s: s.split(sep),
            token_pattern=None,
            stop_words=[''],
            binary=binary,
            dtype=dtype
        ).fit_transform(data)

        return code if norm is None else sklearn.preprocessing.normalize(code, norm=norm)

    # 矩阵，分别编码每列然后拼接
    categories = []
    codes = []
    columns = []

    for i in range(data.shape[1]):
        c = sklearn.feature_extraction.text.CountVectorizer(
            lowercase=False,
            tokenizer=lambda s: s.split(sep),
            token_pattern=None,
            stop_words=[''],
            binary=binary,
            dtype=dtype
        ).fit_transform(data[:, i])

        codes.append(c if norm is None else sklearn.preprocessing.normalize(c, norm=norm))
        columns.append(c.shape[1])

    code = scipy.sparse.hstack(codes)
    # 计算稀疏编码的边界
    limits = numpy.empty(len(columns) + 1, dtype=int)
    limits[0] = 0
    numpy.cumsum(columns, out=limits[1:])
    return code, limits

class OrdinalEncoder(sklearn.preprocessing.OrdinalEncoder):
    """
    修改 `sklearn.preprocessing.OrdinalEncoder` 使未知类别的编码为0.
    """

    def __init__(self, **kwargs):
        super().__init__(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=-1,
            **kwargs
        )

    def transform(self, X):
        return super().transform(X) + 1

    def inverse_transform(self, X):
        return super().inverse_transform(X - 1)

def pc2color(pc):
    """
    根据矩阵主成分分解的结果生成颜色，使样本点的颜色能反映主成分的差异.

    取主成分的前3维变换至 RGB 色系。

    Parameters:
        pc (`numpy.ndarray`): 矩阵分解的主成分

    Returns:
        rgb (`numpy.ndarray`): RGB 颜色值，行数和 pc 相同
    """

    # 缩放传入的主成分使其接近标准正态分布，标准正态分布的3倍标准差区间包含99%以上概率
    rgb = numpy.empty((pc.shape[0], 3), dtype=numpy.float32)
    rgb[:, :pc.shape[1]] = sklearn.preprocessing.StandardScaler() \
        .fit_transform(pc[:, :3]) / 6 + 0.5
    # 如果输入的主成分少于3维，剩余的维度用0.5填充
    rgb[:, pc.shape[1]:] = 0.5

    count = numpy.count_nonzero(numpy.any((rgb < 0) | (rgb > 1), axis=1))
    if count > 0:
        logging.warning(f"{count} points' color out of [0, 1]")

    return numpy.clip(rgb, 0, 1)

def extent(latitudes, longitudes, scale=0, margin=0.01):
    """
    根据样本点坐标计算合适的绘制范围.

    Parameters:
        latitudes (array-like): 样本点纬度
        longitudes (array-like): 样本点经度
        scale (float): 指定绘制范围为样本点的几倍标准差，如不大于0，覆盖所有样本点
        margin (float): 当覆盖所有样本点时，四边的留白

    Returns:
        lat0, lat1, lon0, lon1: 匹配的绘制范围四角坐标
    """

    mask = numpy.logical_and(
        numpy.isfinite(latitudes),
        numpy.isfinite(longitudes)
    )
    latitudes = latitudes[mask]
    longitudes = longitudes[mask]

    # 覆盖所有样本点的最小范围
    ext = numpy.asarray([
        [numpy.min(latitudes), numpy.max(latitudes)],
        [numpy.min(longitudes), numpy.max(longitudes)]
    ])

    if scale > 0:
        # 根据样本点的中心和标准差计算绘制范围
        mean = numpy.asarray([numpy.mean(latitudes), numpy.mean(longitudes)])
        std = numpy.asarray([numpy.std(latitudes), numpy.std(longitudes)])
        # 如果边界超出所有样本点，裁剪
        ext = numpy.clip(
            mean[:, None] + std[:, None] * numpy.asarray([-scale, scale]),
            ext[:, 0:1],
            ext[:, 1:2]
        )

    # 四边添加留白
    ext += (ext[:, 1:2] - ext[:, 0:1]) * numpy.asarray([-margin, margin])
    return ext.flatten()

def clip(func, vmin=0, vmax=1):
    """
    辅助函数，对目标函数的返回值进行截断.
    """

    return lambda x, y: numpy.clip(func(x, y), vmin, vmax)

def make_clip_path(polygons, extent=None):
    """
    根据绘制范围及指定的多边形生成图形的裁剪路径.

    Parameters:
        polygons (`shapely.geometry.multipolygon.MultiPolygon`):
            裁剪的范围，只保留该范围内的图形
        extent: 绘制的范围 (左, 右, 下, 上)

    Returns:
        path (`matplotlib.path.Path`): 生成的裁剪路径，如果传入的多边形为空返回 None
    """

    if polygons is None:
        return None

    polygons = tuple(polygons) if hasattr(polygons, '__iter__') else (polygons,)
    if len(polygons) == 0:
        return None

    if extent is not None:
        # 先对绘图区域和裁剪区域取交集
        xmin, xmax, ymin, ymax = extent
        poly = shapely.geometry.Polygon((
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymax),
            (xmax, ymin),
            (xmin, ymin)
        ))
        polygons = [poly.intersection(c) for c in polygons]

    return matplotlib.path.Path.make_compound_path(
        *cartopy.mpl.patch.geos_to_path(polygons)
    )

def clip_paths(paths, polygons, extent=None):
    """
    根据绘制范围及指定的多边形裁剪 matplotlib 绘制的图形.

    Parameters:
        paths (`matplotlib.PathCollection` or list of `matplotlib.PathCollection`):
            待裁剪的图形
        polygons (`shapely.geometry.multipolygon.MultiPolygon`):
            裁剪的范围，只保留该范围内的图形
        extent: 绘制的范围 (左, 右, 下, 上)
    """

    path = make_clip_path(polygons, extent=extent)
    if path is not None:
        # 裁剪图形
        if hasattr(paths, '__iter__'):
            for c in paths:
                c.set_clip_path(path, transform=c.axes.transData)
        else:
            paths.set_clip_path(path, transform=paths.axes.transData)
