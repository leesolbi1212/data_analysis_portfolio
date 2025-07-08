import pandas as pd

df = pd.read_csv("ev.csv")
# print(df.head())

columns_to_drop = ['노드링크 유형', '노드 ID', '노드 유형 코드','시군구코드','시군구명','읍면동코드','읍면동명','지하철역코드']
df = df.drop(columns=columns_to_drop)
# print(df.info())
df.columns = ['lonlat','sub','exit','empty']
# print(df.info())

df['lon'] = df['lonlat'].str.replace(r'POINT\(', '', regex=True)\
                        .str.replace(r'\)', '', regex=True)\
                        .str.split(' ', expand=True)[0]\
                        .str.replace(',', '')\
                        .astype(float)

df['lat'] = df['lonlat'].str.replace(r'POINT\(', '', regex=True)\
                        .str.replace(r'\)', '', regex=True)\
                        .str.split(' ', expand=True)[1]\
                        .str.replace(',', '')\
                        .astype(float)

df = df.drop(columns='lonlat')

df['exit'] = df['exit'].str.replace('&', ',', regex=False)
# print(df['exit'])

df['exit'] = df['exit'].astype(str) + '번 출구'
# print(df['exit'])
# print(df.info())
# print(df.head())
