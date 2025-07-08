# correlationanalysis.py
# 상관분석
# 두 변수간의 관계를 분석 (주로 연속된 값에 대한 관계)

# economics 데이터
import pandas as pd
economics = pd.read_csv('assets/economics.csv')

# 상관행렬
# 실업자수, 개인소비지출
#print(economics[['unemploy', 'pce']].corr())

# 상관계수는 -1 ~ 1의 값
# 1에 가까울수록 양의 상관, -1에 가까울수록 음의 상관, 0 상관없음
from scipy import stats
# pvalue=np.float64(6.773527303290177e-61) => 6.77... * 10^-61
#print(stats.pearsonr(economics['unemploy'], economics['pce']))

# 상관행렬 히트맵

# 데이터
mtcars = pd.read_csv('assets/mtcars.csv')

# 상관행렬
car_corr = mtcars.corr()
#print(car_corr)

# 소수점 두째자리 반올림
car_corr = round(car_corr, 2)
#print(car_corr)

# 그래프 설정
import matplotlib.pyplot as plt
plt.rcParams.update({
    'figure.dpi': '120',
    'figure.figsize': [7.5, 5.5]
})

# 히트맵
import seaborn as sns
# 인자 : 상관행렬, 상관계수표시, 컬러맵
#sns.heatmap(car_corr, annot=True, cmap='RdBu')
#plt.show()

# 대각행렬 제거용 mask만들기
import numpy as np
mask = np.zeros_like(car_corr) # 값이 모두 0인 행렬
#print(mask)

# 오른쪽 위 대각행렬을 1로 변경
mask[np.triu_indices_from(mask)] = 1
#print(mask)

# mask 적용된 히트맵
# sns.heatmap(data=car_corr, annot=True, cmap='RdBu', mask=mask)
# plt.show()

# 빈 행, 빈 열 히트맵에서 제거
mask_new = mask[1:, :-1]
cor_new = car_corr.iloc[1:, :-1]

# 최종 히트맵
#sns.heatmap(data=cor_new, annot=True, cmap='RdBu', mask=mask_new)
#plt.show()

# 히트맵 설정 옵션
sns.heatmap(
    data = cor_new, # 상관행렬
    annot = True, # 상관계수 표시
    cmap = 'RdBu', # 컬러맵
    mask = mask_new, # 마스크
    linewidths = .5, # 경계선 두께
    vmax = 1, # 가장 진한 파란색으로 표현할 최대값
    vmin = -1, # 가장 진한 빨간색으로 표현할 최대값
    cbar_kws = {'shrink': .5} # 범례 크기
)
plt.show()





































