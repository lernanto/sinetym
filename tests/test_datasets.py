# -*- coding: utf-8 -*-

"""
测试 SinComp 数据集相关功能
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import pytest
import os
import pandas


@pytest.fixture(scope='session')
def datasets(tmp_path_factory):
    """为缓存文件创建临时目录，为数据集设置环境变量"""

    # 设置测试必须的环境变量
    os.environ['SINCOMP_CACHE'] = str(tmp_path_factory.mktemp('sincomp'))
    os.environ['ZHONGGUOYUYAN_HOME'] = os.path.join(
        os.path.dirname(__file__),
        'data',
        'zhongguoyuyan'
    )

    import sincomp.datasets
    return sincomp.datasets

@pytest.fixture(scope='class')
def custom_dataset(datasets):
    """加载自定义文件数据集"""

    return datasets.FileDataset(path=os.path.join(
        os.path.dirname(__file__),
        'data',
        'custom_dataset'
    ))


class TestFileDataset:
    def test_data(self, custom_dataset):
        data = custom_dataset.data
        assert isinstance(data, pandas.DataFrame)
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            assert col in data.columns

    def test_filter(self, custom_dataset):
        data = custom_dataset.filter(['01']).data
        assert isinstance(data, pandas.DataFrame)
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            assert col in data.columns

    def test_sample(self, custom_dataset):
        data = custom_dataset.sample(n=1).data
        assert isinstance(data, pandas.DataFrame)
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            assert col in data.columns

    def test_shuffle(self, custom_dataset):
        data = custom_dataset.shuffle().data
        assert isinstance(data, pandas.DataFrame)
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            assert col in data.columns


class TestCCRDataset:
    def test_dialect_info(self, datasets):
        info = datasets.ccr.metadata['dialect_info']
        assert isinstance(info, pandas.DataFrame)
        for col in 'group', 'subgroup', 'cluster', 'subcluster', 'spot':
            assert col in info.columns

    def test_char_info(self, datasets):
        info = datasets.ccr.metadata['char_info']
        assert isinstance(info, pandas.DataFrame)

    def test_load_data(self, datasets):
        _, data = datasets.ccr.load_data('C027')[0]
        assert isinstance(data, pandas.DataFrame)
        for col in 'did', 'cid', 'character', 'initial', 'final', 'tone':
            assert col in data.columns


class TestMCPDictDataset:
    def test_dialect_info(self, datasets):
        info = datasets.mcpdict.metadata['dialect_info']
        assert isinstance(info, pandas.DataFrame)
        for col in 'group', 'subgroup', 'cluster', 'subcluster', 'spot':
            assert col in info.columns

    def test_load_data(self, datasets):
        _, data = datasets.mcpdict.load_data('MB1-002')[0]
        assert isinstance(data, pandas.DataFrame)
        for col in 'did', 'character', 'initial', 'final', 'tone':
            assert col in data.columns


class TestZhongguoyuyanDataset:
    def test_dialect_info(self, datasets):
        info = datasets.zhongguoyuyan.metadata['dialect_info']
        assert isinstance(info, pandas.DataFrame)
        for col in 'group', 'subgroup', 'cluster', 'subcluster', 'spot':
            assert col in info.columns

    def test_char_info(self, datasets):
        info = datasets.zhongguoyuyan.metadata['char_info']
        assert isinstance(info, pandas.DataFrame)

    def test_load_data(self, datasets):
        _, data = datasets.zhongguoyuyan.load_data('Z06K06')[0]
        assert isinstance(data, pandas.DataFrame)
        for col in 'did', 'character', 'initial', 'final', 'tone':
            assert col in data.columns
