# dataframe.py
# pandas 라이브러리의 DataFrame

# pandas 라이브러리 임포트
import pandas as pd

# 딕셔너리로 DataFrame 생성
df = pd.DataFrame({
    'name': ['홍길동', '강감찬', '이순신'],
    'kor': [90, 80, 70],
    'eng': [100, 90, 80],
    'math': [60, 50, 40]
})

# DataFrame 출력
print(df)
print(type(df))

# name변수 출력
# 출력결과 : Series
# Series는 1차원(선형) 데이터에 인덱스가 있는 형태
# DataFrame은 2차원(행렬) 데이터에 행/열인덱스가 있는 형태
# Series가 2개 이상 모이면 DataFrame
print(df['name'])
print(type(df['name']))

# kor변수의 합 출력
print(sum(df['kor']))

# 엑셀파일로 DataFrame 생성 후 출력
# openpyxl 외부라이브러리 필요
df_xl_exam = pd.read_excel('assets/exam.xlsx')
print(df_xl_exam)

# 행의 수
print(len(df_xl_exam))

# 열(변수)의 수
print(df_xl_exam.shape[1])

# CSV파일로 DataFrame 생성
df_csv_exam = pd.read_csv('assets/exam.csv')
print(df_csv_exam)

# DataFrame을 CSV파일로
# index=False : 행인덱스 무시
df_csv_exam.to_csv('assets/exam2.csv', index=False)

# JSON파일로 DataFrame 생성
df_json_exam = pd.read_json('assets/exam.json')
print(df_json_exam)

# DataFrame을 JSON파일로
# index=4 : 들여쓰기 스페이스 4개
df_json_exam.to_json('assets/exam2.json', indent=4)

# XML파일로 DataFrame 생성
# lxml 외부라이브러리 필요
df_xml_exam = pd.read_xml('assets/exam.xml')
print(df_xml_exam)

# DataFrame을 XML파일로
df_xml_exam.to_xml('assets/exam2.xml', index=False)















