# datacleaning.py
# 데이터 정제

# 라이브러리 임포트
import pandas as pd
import numpy as np

# 결측치 (missing value)
# np.nan : 데이터가 없음 = 결측치
df = pd.DataFrame({
    'name': ['홍길동', '강감찬', np.nan, '이순신'],
    'score': [100, 90, 80, np.nan]
})
#print(df)

# 결측치가 있는 상태에서의 연산
# NaN과 어떤 연산을 하던지 결과는 NaN, NaN은 연산 불가
df['compscore'] = df['score'] - 10
#print(df)

# 데이터프레임내의 결측치 확인
# 결측치인 경우 True로 표시
# print(pd.isna(df))

# 결측치 개수 확인
#print(pd.isna(df).sum())

# score 결측치 제거
# 결측치가 있는 행전체를 제거
# print(df.dropna(subset=['score']))

# 전체 결측치 제거
# print(df.dropna(subset=['name', 'score']))

# 결측치 대체(impultation)
# print(df)
# # loc[행인덱스리스트, 열인덱스리스트]
# df.loc[[2], ['name']] = '유관순'
# print(df)
# df.loc[[3], ['score']] = 70
# print(df)
# # compscore의 모든 결측치를 80으로 대체
# df['compscore'] = df['compscore'].fillna(80)
# print(df)

# 이상치 (anormaly value)
# 데이터 분석시 일반적으로 제외해야할 데이터
# 도봉구를 지역의 명칭으로 가정하고, 점수는 100점까지라고 가정하면
# 도봉구와 120은 이상치
df2 = pd.DataFrame({
    'name': ['홍길동', '강감찬', '도봉구'],
    'score': [100, 120, 80]
})

# 이상치를 결측치로 처리
df2['score'] = np.where(df2['score']>100, np.nan, df2['score'])
# print(df2)

# 결측치로 처리한 이상치를 제거
df2_result = df2.dropna(subset=['score'])
# print(df2_result)

# 극단치 (outlier value)
# 이상치의 일종으로 논리적으로 존재할 수는 있지만
# 극단적으로 크거나 작은 값으로 전체 분석에 극단적인 영향을
# 미칠 수 있으므로 데이터 정제 정책에 따라서
# 분석에 포함시키거나 배제할 수 있음

# 그래프 라이브러리인 seaborn 라이브러리 필요
# mpg.csv 파일 사용
import seaborn as sns
import matplotlib.pyplot as plt
mpg = pd.read_csv('assets/mpg.csv')
# 데이터프레임, Y축데이터
sns.boxplot(data=mpg, y='hwy')
#plt.show()

# 극단치 기준값 (Q1, Q3)
pct25 = mpg['hwy'].quantile(.25) # 1사분위수, 25%
pct75 = mpg['hwy'].quantile(.75) # 3사분위수, 75%
#print(pct25, pct75)

# IQR (Inter Quantile Range:사분위 범위)
# : 1사분위수와 3사분위수 간의 거리(범위)
# IQR = 3사분위수 - 1사분위수
iqr = pct75 - pct25
#print(iqr)

# 상한, 하한 정하기
uplimit = pct75 + iqr*1.5 # 27 + 13.5 = 40.5
downlimit = pct25 - iqr*1.5 # 18 - 13.5 = 4.5
#print(uplimit, downlimit)

# 상한, 하한 기준으로 극단치를 결측치 처리
mpg['hwy'] = np.where(
    (mpg['hwy']<4.5) | (mpg['hwy']>40.5),
    np.nan,
    mpg['hwy']
)
# print(mpg['hwy'].isna().sum()) # 3개가 결측치 처리됨

# 결측치 제거
mpg = mpg.dropna(subset=['hwy'])
#print(mpg)

# 구동방식별 고속도로연비(hwy)의 평균
#print(mpg.groupby('drv').agg(mean_hwy=('hwy', 'mean')))

































