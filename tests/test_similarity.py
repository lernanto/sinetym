# -*- coding: utf-8 -*-

"""
测试计算方言之间相似度
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import pytest
import os
import pandas
import numpy
import sincomp.preprocess
import sincomp.similarity

from test_datasets import datasets


@pytest.fixture(scope='module')
def data(datasets):
    """加载测试用数据集并变换成需要的宽表格式"""

    return sincomp.preprocess.transform(
        datasets.FileDataset(path=os.path.join(
            os.path.dirname(__file__),
            'data',
            'custom_dataset'
        )),
        index='cid',
        values=['initial', 'final', 'tone'],
        aggfunc='first'
    )


def test_chi2(data):
    sim = sincomp.similarity.chi2(data)
    assert sim.shape == (data.columns.levels[0].shape[0],) * 2
    assert numpy.all(numpy.isfinite(sim))

def test_entropy(data):
    sim = sincomp.similarity.entropy(data)
    assert sim.shape == (data.columns.levels[0].shape[0],) * 2
    assert numpy.all(numpy.isfinite(sim))
