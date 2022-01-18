#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

'''
爬取中国语言资源保护工程采录展示平台数据

@see https://zhongguoyuyan.cn
'''

__author__ = '黄艺华 <lernanto@foxmail.com>'


import sys
import logging
import os
import requests
import uuid
import ddddocr
import json
import pandas
import time


site = 'https://zhongguoyuyan.cn'
verify_code_url = '{}/svc/common/create/verifyCode'.format(site)
login_url = '{}/svc/user/login'.format(site)
logout_url = '{}/svc/user/logout'.format(site)
logged_url = '{}/svc/common/validate/logged'.format(site)
survey_url = '{}/svc/mongo/query/latestSurveyMongo'.format(site)
data_url = '{}/svc/mongo/resource/normal'.format(site)


def get_verify_code(session):
    '''请求验证码'''

    logging.info('get verify code from {}'.format(verify_code_url))
    rsp = session.get(verify_code_url)
    logging.debug('cookies = {}'.format(session.cookies))

    return rsp.content

def get_token(image, ocr):
    '''从图片识别验证码'''

    # 使用 OCR 识别图片中的验证码
    token = ocr.classification(image)
    logging.info('get token from image, token = {}'.format(token))
    return token

def login(session, email, password, token):
    '''登录到网站'''

    # 发送登录请求
    logging.info('login from {}, email = {}, pass = {}, token = {}'.format(
        login_url,
        email,
        password,
        token
    ))
    rsp = session.post(
        login_url,
        data={'email': email, 'pass': password, 'token': token}
    )
    ret = rsp.json()
    logging.debug(ret)
    logging.debug('cookies = {}'.format(session.cookies))

    code = ret.get('code')
    if code == 200:
        # 登录成功
        logging.info('login successful, code = {}'.format(code))
    else:
        # 登录失败
        logging.error('login fialed, code = {}, {}'.format(
            code,
            ret.get('description')
        ))

    return code

def logout(session):
    '''退出登录'''

    logging.info('logout from {}'.format(logout_url))
    session.get(logout_url)

def validate_login(session):
    '''验证登录状态'''

    # 验证是否登录成功
    logging.info('validate login from {}'.format(logged_url))
    rsp = session.post(logged_url)
    ret = rsp.json()
    logging.debug(ret)
    logging.debug('cookies = {}'.format(session.cookies))

    if ret.get('loginFlag') == 1:
        # 用户已经登录，输出用户信息
        user = ret.get('user', {})
        logging.info('user logged in, user ID = {}, E-mail = {}, last logged in at {}'.format(
            user.get('user_id'),
            user.get('email'),
            user.get('login_record', [{}])[-1].get('time')
        ))
        return True
    else:
        # 用户未登录
        logging.info('user not logged in')
        return False

def get_survey(session):
    '''获取全部调查点'''

    logging.info('get survey from {}'.format(survey_url))
    rsp = session.post(survey_url)
    data = rsp.json()
    logging.debug(data)

    status = data.get('status')
    if status == 'success':
        logging.info('get survey sucessful')
    else:
        logging.error('get survey failed, status = {}'.format(status))

    return data

def get_dialect(session, id):
    '''获取方言点数据'''

    referer = '{}/area_details.html?id={}'.format(site, id)

    logging.info('get dialect data from {}, ID = {}'.format(data_url, id))
    rsp = session.post(data_url, headers={'referer': referer}, data={'id': id})
    ret = rsp.json()
    logging.debug(ret)

    code = ret.get('code')
    if code == 200:
        # 获取数据成功
        logging.debug('get data successful, code = {}'.format(code))
        return ret.get('data')
    else:
        logging.error('get data failed, code = {}, {}'.format(
            code,
            ret.get('description')
        ))

def try_login(email, password, retry=3):
    '''尝试登录，如果验证码错误，重试多次'''

    session = requests.Session()
    # 网站要求客户端生成一个 UUID 唯一标识当前会话
    session.cookies.set('uniqueVisitorId', str(uuid.uuid4()))

    # 用于识别验证码
    ocr = ddddocr.DdddOcr()

    for i in range(retry):
        # 尝试登录
        image = get_verify_code(session)
        token = get_token(image, ocr)
        code = login(session, email, password, token)

        # 记录一下验证码图片和识别结果
        dir = 'verify_code'
        os.makedirs(dir, exist_ok=True)
        fname = os.path.join(
            dir,
            '.'.join(('_'.join((
                'zhongguoyuyan',
                uuid.uuid4().hex,
                token,
                '0' if code == 702 else '1'
            )), 'jpg'))
        )
        logging.info('save verify code to {}'.format(fname))
        with open(fname, 'wb') as f:
            f.write(image)

        if code != 702:
            # 不是验证码错误，无论登录成功失败都跳出
            break

    if code == 200:
        logging.info('login sucessfule after {} try'.format(i + 1))
        return session
    else:
        logging.error('login failed after {} try, give up'.format(i + 1))

def parse_survey(survey):
    '''解析调查点 JSON 数据，转换成表格'''

    objects = []
    for obj in ('dialectObj', 'minorityObj'):
        data = pandas.json_normalize(survey[obj], 'cityList', 'provinceCode')
        data.insert(1, 'proviceCode', data.pop('provinceCode'))
        objects.append(data)

    return objects

def parse_dialect(dialect):
    '''解析方言 JSON 数据，转换成表格'''

    resources = {}
    for res in dialect.get('resourceList', []):
        oid = res['items'][0]['oid']
        record = pandas.json_normalize(res.get('items'), 'records', ['iid', 'name'])
        record.insert(0, 'iid', record.pop('iid'))
        resources[oid] = record

    return resources

def crawl_survey(session, prefix='.'):
    '''爬取调查点列表并保存到文件'''

    # 获取调查点列表
    survey = get_survey(session)
    fname = os.path.join(prefix, 'survey.json')
    logging.info('save survey data to {}'.format(fname))
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(survey, f, ensure_ascii=False, indent=4)

    # 分别保存方言列表和少数民族语言列表
    dialect, minority = parse_survey(survey)
    dialect_file = os.path.join(prefix, 'dialect.csv')
    logging.info('save dialect list to {}'.format(dialect_file))
    dialect.to_csv(dialect_file, index=False)
    minority_file = os.path.join(prefix, 'minority.csv')
    logging.info('save minority list to {}'.format(minority_file))
    minority.to_csv(minority_file, index=False)

    return dialect, minority

def crawl_dialect(session, id, prefix='.'):
    '''爬取一个方言点的数据并保存到文件'''

    dialect_file = os.path.join(prefix, '{}.json'.format(id))
    if os.path.exists(dialect_file):
        # 数据文件已存在，认为之前已经爬取过，不再重复爬取
        logging.info(
            'dialect data file {} already exists, do nothing'.format(dialect_file)
        )
        return None

    # 请求方言数据
    data = get_dialect(session, id)

    if data:
        # 保存原始数据
        logging.info('save dialect data to {}'.format(dialect_file))
        with open(dialect_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        # 把单字音、词汇等数据分别保存成 CSV 文件
        resources = parse_dialect(data)
        for oid, res in resources.items():
            fname = os.path.join(prefix, '{}_{}.csv'.format(id, oid))
            logging.info('save resource to {}'.format(fname))
            res.to_csv(fname, encoding='utf-8', index=False)

        return True
    else:
        return False


def main():
    logging.getLogger().setLevel(logging.INFO)

    email, password, prefix, max_crawl = sys.argv[1:5]
    max_crawl = int(max_crawl)
    crawl = 0

    # 登录网站
    session = try_login(email, password)
    if session:
        # 爬取调查点列表
        dialect, minority = crawl_survey(session, prefix)

        # 当前只处理方言数据
        for _, row in dialect.iterrows():
            # 爬取一个方言调查点的数据
            id = row['_id']
            logging.info('crawl dialect ID = {}, city = {}'.format(id, row['city']))
            ret = crawl_dialect(session, id, prefix)

            # 无论爬取成功还是失败，都算一次
            crawl += (0 if ret is None else 1)
            if crawl >= max_crawl:
                logging.info('reach maximum crawl number = {}, have a rest'.format(crawl))
                break

            # 延迟一段时间，降低吞吐量
            time.sleep(1)

        if crawl < max_crawl:
            logging.info('all data crawled, nothing else todo, crawl = {}'.format(crawl))

        # 退出登录
        logout(session)

    logging.info('totally crawl {} data'.format(crawl))

if __name__ == '__main__':
    main()