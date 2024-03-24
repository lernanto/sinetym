# -*- coding: utf-8 -*-

"""
测试预处理方言读音数据函数
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import pytest
import os
import pandas
import sincomp.preprocess


@pytest.fixture(scope='module')
def data():
    """加载测试用数据"""

    return pandas.concat(
        [pandas.read_csv(e.path, dtype=str, encoding='utf-8') \
            for e in os.scandir(os.path.join(
                os.path.dirname(__file__),
                'data',
                'custom_dataset'
            )) if e.is_file()],
        axis=0,
        ignore_index=True
    )

def test_clean_ipa(data):
    clean = sincomp.preprocess.clean_ipa(data['initial'])
    assert clean.shape == data['initial'].shape

def test_clean_initial(data):
    clean = sincomp.preprocess.clean_initial(data['initial'])
    assert clean.shape == data['initial'].shape

def test_clean_final(data):
    clean = sincomp.preprocess.clean_final(data['final'])
    assert clean.shape == data['final'].shape

def test_clean_tone(data):
    clean = sincomp.preprocess.clean_tone(data['tone'])
    assert clean.shape == data['tone'].shape

def test_transform(data):
    output = sincomp.preprocess.transform(data, index='cid')
    assert output.shape[0] == data['cid'].value_counts().shape[0]
    assert output.notna().any(axis=0).all()
