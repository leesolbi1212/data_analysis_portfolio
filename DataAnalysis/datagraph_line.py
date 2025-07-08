# datagraph_line.py
# 라인 그래프
# 시계열 데이터(시간의 흐름에 따라 변화하는 데이터) 표현

# 라이브러리 임포트
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터프레임
economics = pd.read_csv('assets/economics.csv')
#economics.info()

# 라인 그래프
#sns.lineplot(data=economics, x='date', y='unemploy')
#plt.show()

# x축에 연도 표시
economics['date2'] = pd.to_datetime(economics['date']) # 날짜타입으로 변환
economics['year'] = economics['date2'].dt.year # 연도
# sns.lineplot(data=economics, x='year', y='unemploy')
# plt.show()

# 신뢰구간 표시 제거
# ci=None : 신뢰구간 표시 제거
sns.lineplot(data=economics, x='year', y='unemploy', ci=None)
plt.show()































