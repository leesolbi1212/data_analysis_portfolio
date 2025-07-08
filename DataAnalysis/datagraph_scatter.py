# datagraph_scatter.py
# 산점도 그래프
# 데이터를 X/Y축에 표시
# 두 변수간의 관계를 표현

# 라이브러리 임포트
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 로딩
mpg = pd.read_csv('assets/mpg.csv')
#mpg.info()

# 산점도 그래프
sns.scatterplot(data=mpg, x='displ', y='hwy') # 배기량, 고속도로연비
#plt.show()

# 산점도 그래프 축 범위 설정
sns.scatterplot(data=mpg, x='displ', y='hwy') \
    .set(xlim=[3, 6])
#plt.show()
sns.scatterplot(data=mpg, x='displ', y='hwy') \
    .set(xlim=[3, 6], ylim=[10, 30])
#plt.show()

# 분류별 색상 변경
sns.scatterplot(data=mpg, x='displ', y='hwy', hue='drv') # hue : 범례, 분류
#plt.show()

# 그래프 기본 설정
plt.rcParams.update({'figure.dpi': '150'}) # 해상도, 기본값 72
plt.rcParams.update({'figure.figsize': [8, 6]}) # 가로/세로 크기, 기본값 [6, 4]
plt.rcParams.update({'font.size': '15'}) # 글자 크기, 기본값 10
plt.rcParams.update({'font.family': 'Malgun Gothic'}) # 글자체, 기본값 sans-serif
plt.show()





















