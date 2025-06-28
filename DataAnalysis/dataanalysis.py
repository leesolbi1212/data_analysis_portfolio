# dataanalysis.py
# Pandas 데이터 분석

import pandas as pd

exam = pd.read_csv('assets/exam3.csv')
#print(exam)

# 상위 5행
#print(exam.head())

# 상위 10행
# print(exam.head(10))

# 하위 5행
#print(exam.tail())

# 하위 10행
#print(exam.tail(10))

# 행의 수, 열의 수
# 튜플내에 숫자가 2개 있으면 2차원, 3개 있으면 3차원
#print(exam.shape)

# 데이터프레임 전체 구조
#exam.info()

# 요약 통계 (기초통계)
# 숫자열에 대해서 개수,평균,표준편차,최소,25%,50%,75%,최대
#print(exam.describe())

# DataFrame 생성
df_org = pd.DataFrame({
    'var1': [1, 2, 1],
    'var2': [2, 3, 2]
})

# 변수명 변경
df_new = df_org.rename(columns={'var1': 'data1'})
# print(df_new)
# print(df_org) # 원래 데이터프레임은 변경 안됨

# 새로운 변수 (파생변수) 생성
df_new['data2'] = df_new['data1'] + df_new['var2']
#print(df_new)

# 그래프 출력을 위해 matplotlib 외부라이브러리 필요

# 그래프 라이브러리
import matplotlib.pyplot as plt

# 그래프에 표시할 데이터프레임
df_score = pd.DataFrame({
    'name': ['홍길동', '강감찬', '이순신'],
    'kor': [90, 80, 70],
    'eng': [100, 90, 80],
    'math': [60, 50, 40]
})

# 데이터프레임에서 그래프 출력
# 박스플랏 (상자그래프) : 데이터의 분포 확인
df_score.boxplot(column=['kor', 'eng', 'math'])
# 그래프를 화면에 출력
#plt.show()

# numpy는 파이썬에 없는 배열 처리 및 수치연산 하기 위한 라이브러리
import numpy as np

# numpy를 활용한 DataFrame 조건 부여
# 수학점수가 60점 이상이면 test변수는 pass의 값을 가지고
# 그렇지 않으면 fail의 값을 가짐
df_score['test'] = np.where(df_score['math']>=60, 'pass', 'fail')
#print(df_score)

# 조건에 따른 pass, fail 개수
# value_counts() : 컬럼 값을 그룹핑해서 그룹내 값의 개수를 반환
counts = df_score['test'].value_counts()
#print(counts)

# 조건 중첩
df_score['grade'] = np.where(df_score['math']>=60, 'A',
                             np.where(df_score['math']>=50, 'B', 'C'))
#print(df_score)

# 수학점수에 따른 오름차순 정렬
df_score_sorted = df_score['math'].sort_values()
# print(df_score_sorted)

# 수학점수에 따른 내림차순 정렬
df_score_sorted = df_score['math'].sort_values(ascending=False)
# print(df_score_sorted)

















































