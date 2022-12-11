#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

"""
使用 D-Tale 创建服务展示方言数据.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import argparse
import logging
import os
import pandas as pd
import dtale

import sinetym
from sinetym.datasets import zhongguoyuyan


def combine_phone(part1, part2):
    """连接字音数据表中声韵调的辅助函数."""

    return part1.combine(part2, (lambda x, y: [i + j for (i, j) in zip(x, y)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('-H', '--host', default='localhost', help='监听的地址或主机名')
    parser.add_argument('-p', '--port', type=int, default=40000, help='监听的端口')
    parser.add_argument('path', help='方言数据根目录路径')
    args = parser.parse_args()

    dialect_path = os.path.join(args.path, 'dialect')
    location = zhongguoyuyan.load_location(
        os.path.join(dialect_path, 'location.csv')
    )
    char = pd.read_csv(os.path.join(args.path, 'words.csv'), index_col=0)
    data = sinetym.datasets.transform_data(
        zhongguoyuyan.load_data(dialect_path),
        index='cid',
        agg=' '.join
    ).reindex(char.index)

    location[['area', 'slice', 'slices']] \
        = location[['area', 'slice', 'slices']].fillna('')
    dtale.views.startup(
        data=location[[
            'province',
            'city',
            'county',
            'dialect',
            'area',
            'slice',
            'slices',
            'latitude',
            'longitude'
        ]],
        name='location',
        locked=['city', 'county']
    )

    # 字表及代表方言点的读音
    indeces = [
        '03E88',    # 北京
        '10G68',    # 南京
        '12177',    # 太原
        '10G71',    # 苏州
        '08210',    # 温州
        '21J03',    # 黄山
        '26516',    # 长沙
        '26509',    # 双峰
        '18358',    # 南昌
        '15233',    # 梅州
        '15231',    # 广州
        '01G23',    # 南宁
        '02G49',    # 厦门
        '02193'     # 建瓯
    ]
    columns = ['initial', 'final', 'tone']
    dtale.views.startup(
        data=pd.DataFrame(
            data.loc[:, (indeces, columns)] \
                .applymap(lambda x: ' '.join(sorted(set(x.split())))).values,
            index=char.index.astype(str) + char['item'],
            columns=pd.MultiIndex.from_product((
                location.loc[indeces, ['city', 'county']].apply(''.join, axis=1),
                columns
            ))
        ),
        name='character',
        inplace=True
    )

    # 拼接声韵调成为完整读音
    data = data.apply(lambda c: c.str.split(' '))
    pronunciation = data.loc[:, pd.IndexSlice[:, 'initial']] \
        .droplevel(axis=1, level=1).combine(
        data.loc[:, pd.IndexSlice[:, 'final']].droplevel(axis=1, level=1),
        combine_phone
    ).combine(
        data.loc[:, pd.IndexSlice[:, 'tone']].droplevel(axis=1, level=1),
        combine_phone
    ).combine(
        data.loc[:, pd.IndexSlice[:, 'memo']].droplevel(axis=1, level=1),
        combine_phone
    ).apply(lambda c: c.str.join(' ').str.replace('∅', ''))

    pronunciation.columns = pronunciation.columns \
        + location.loc[pronunciation.columns, 'city'] \
        + location.loc[pronunciation.columns, 'county']
    pronunciation.set_index(char.index.astype(str) + char['item'], inplace=True)

    dtale.views.startup(
        data=pronunciation,
        name='pronunciation',
        inplace=True,
        show_columns=['index'] + (indeces + location.loc[indeces, 'city'] \
            + location.loc[indeces,'county']).tolist()
    )

    # 启动 D-Tale 服务
    dtale.app.build_app().run(host=args.host, port=args.port)