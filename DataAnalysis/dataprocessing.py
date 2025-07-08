# dataprocessing.py
# 데이터 전처리

import pandas as pd

df = pd.DataFrame({
    'name': ['kim', 'lee', 'park', 'choi', 'hong'],
    'nclass': [1, 1, 2, 2, 2],
    'kor': [10, 20, 30, 40, 50],
    'eng': [30, 40, 50, 60, 70],
    'math': [50, 60, 70, 80, 90]
})
#print(df)
#df.info()

# 조건에 맞는 행
# print(df.query('kor==10'))
# print(df.query('kor!=10'))
# print(df.query('eng>50'))
# print(df.query('nclass==1 & eng>=40'))
# print(df.query('nclass==1 | eng>=40'))
# print(df.query('kor in [10, 30, 50]'))

# 변수를 조건에 사용
korList = [10, 30, 50]
#print(df.query('kor in @korList'))

# 특정 변수만 추출
#print(df['kor'])
#print(type(df['kor'])) # Series
#print(df[['eng', 'math']])
#print(type(df[['eng', 'math']])) # DataFrame

# 변수 제거
#print(df.drop(columns='math'))
#print(df) # 원래 DataFrame은 변경되지 않음

# 함수 조합
# print(df.query('nclass==2')[['kor']])
# print(df.query('nclass==2')[['nclass', 'kor']])
# print(df.query('nclass==2')[['kor']].head(2))

# 데이터 정렬
# print(df)
# 값으로 오름차순 정렬
# print(df.sort_values('kor'))
# 값으로 내림차순 정렬
# print(df.sort_values('kor', ascending=False))
# 인덱스로 오름차순 정렬
# print(df.sort_index())
# 인덱스로 내림차순 정렬
# print(df.sort_index(ascending=False))

# 새로운 변수 (파생변수) 추가
totaldf = df.assign(total = df['kor'] + df['eng'] + df['math'])
#df['totaldf'] = df['kor'] + df['eng'] + df['math']
#print(totaldf)

# 파생변수 추가시 조건 부여
# import numpy as np
# meandf = totaldf.assign(mean = totaldf['total']/3)
# resultdf = meandf.assign \
#     (result = np.where(meandf['mean']>=60, 'pass', 'fail'))
# print(resultdf)

# 그룹핑

# 전체 수학 평균
# agg : 집계함수
# mean_math 변수는 수학 평균 값을 가지는 변수
# print(df.agg(mean_math = ('math', 'mean')))

# 반별 수학 평균
# groupby : 값이 같은 것들을 묶음(그룹핑)
# as_index=False : nclass를 인덱스로 사용하지 않고 일반 열로 사용
# print(df.groupby('nclass', as_index=False) \
#     .agg(mean_math=('math', 'mean')))

# 반별 여러 통계
# print(df.groupby('nclass', as_index=False).agg(
#     mean_math = ('math', 'mean'), # 수학 평균
#     sum_math = ('math', 'sum'), # 수학 합계
#     median_math = ('math', 'median'), # 수학 중간값
#     count_math = ('math', 'count') # 수학 점수의 개수
# ))

# 데이터 합치기
df1 = pd.DataFrame({
    'id': [1, 2, 3], # 아이디
    'mid': [100, 90, 80] # 중간고사 점수
})
df2 = pd.DataFrame({
    'id': [1, 2, 4], # 아이디
    'final': [90, 80, 70] # 기말고사 점수
})

# 행 합치기
# print(pd.concat([df1, df2]))
# print(pd.concat([df2, df1]))

# 열 합치기
# how : 1. inner : 양쪽 모두에 있는 키만 병합 (교집합)
#       2. left : df1에 있는 행을 기준으로 일치하는 데이터만 df2에서 가져옴
#       3. right : df2에 있는 행을 기준으로 일치하는 데이터만 df1에서 가져옴
#       4. outer : 양쪽 모두의 키를 포함하고, 일치하지 않으면 NaN (합집합)
# on : 키
print(pd.merge(df1, df2, how='inner', on='id'))
print(pd.merge(df1, df2, how='left', on='id'))
print(pd.merge(df1, df2, how='right', on='id'))
print(pd.merge(df1, df2, how='outer', on='id'))































