# kaggledata.py
import aiohttp
import aiofiles
import asyncio
import os
import pandas as pd
import json
import zipfile

# Kaggle 인증 정보 변수
KAGGLE_USERNAME = ''
KAGGLE_KEY = ''

# 비동기 인증 함수
async def load_kaggle_credentials():
    global KAGGLE_USERNAME, KAGGLE_KEY
    kaggle_path = os.path.expanduser('~/.kaggle/kaggle.json')
    if os.path.exists(kaggle_path):
        with open(kaggle_path, 'r') as f:
            cred = json.load(f)
            KAGGLE_USERNAME = cred['username']
            KAGGLE_KEY = cred['key']
    else:
        raise Exception('kaggle.json 파일이 필요합니다!')


# 다운로드 함수
async def download_file(session, url, save_path):
    async with session.get(url) as response:
        if response.status == 200:
            async with aiofiles.open(save_path, mode='wb') as f:
                await f.write(await response.read())
            print(f'다운로드 완료 : {save_path}')
        else:
            raise Exception(f'다운로드 실패 : {response.status}')


# 메인 함수
async def main():
    await load_kaggle_credentials()

    dataset_owner = 'zynicide'
    dataset_name = 'wine-reviews'
    # zip 파일로 다운로드
    kaggle_url = f'https://{KAGGLE_USERNAME}:{KAGGLE_KEY}@www.kaggle.com/api/v1/datasets/download/{dataset_owner}/{dataset_name}'

    zip_path = './wine_reviews.zip'
    extract_path = './wine_reviews'

    # 다운로드
    async with aiohttp.ClientSession() as session:
        await download_file(session, kaggle_url, zip_path)

    # 압축풀기
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    print('압축 해제 완료')

    # 추출된 csv 파일 읽기
    csv_file = os.path.join(extract_path, 'winemag-data_first150k.csv')
    if not os.path.exists(csv_file):
        csv_file = os.path.join(extract_path, 'winemag-data-130k-v2.csv')  # 다른 버전 파일명 대응

    df = pd.read_csv(csv_file, encoding='utf-8')
    print(df.head())


if __name__ == '__main__':
    asyncio.run(main())
