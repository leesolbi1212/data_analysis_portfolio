import pandas as pd

df = pd.read_csv("ev_2.csv")

df['comment'] = df['sub'].astype(str) + ' ' + df['exit'].astype(str)

# 다중 INSERT SQL 구문 생성
sql_list = []
for _, row in df.iterrows():
    type_val = 3
    lon = row['lon']
    lat = row['lat']
    address = ''
    comment = row['comment'].replace("'", "''")  # 작은따옴표 이스케이프
    weight = 0

    sql = f"({type_val}, {lon}, {lat}, '{address}', '{comment}', {weight})"
    sql_list.append(sql)

# 전체 INSERT 문 구성
insert_sql = 'INSERT INTO marker (type, lon, lat, address, comment, weight)\nVALUES\n' + ',\n'.join(sql_list) + ';'

# 파일로 저장
sql_file_path = 'marker_insert.sql'
with open(sql_file_path, 'w', encoding='utf-8') as f:
    f.write(insert_sql)

sql_file_path