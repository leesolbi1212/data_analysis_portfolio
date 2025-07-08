# datagraph_bar.py
# 막대 그래프
# 데이터간의 차이

# 라이브러리 임포트
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터프레임
mpg = pd.read_csv('assets/mpg.csv')

# 막대 그래프용 데이터
# 구동방식별로 그룹핑한 후 고속도로 연비의 평균
df_mpg = mpg.groupby('drv', as_index=False) \
    .agg(mean_hwy=('hwy', 'mean'))
#print(df_mpg)

# 막대 그래프
#sns.barplot(data=df_mpg, x='drv', y='mean_hwy')
#plt.show()

# 막대 크기순 정렬
df_mpg = df_mpg.sort_values('mean_hwy', ascending=False)
# sns.barplot(data=df_mpg, x='drv', y='mean_hwy')
# plt.show()

# 빈도 막대그래프 1
df_mpg = mpg.groupby('drv', as_index=False) \
    .agg(n=('drv', 'count'))
# sns.barplot(data=df_mpg, x='drv', y='n')
# plt.show()

# 빈도 막대그래프 2
# sns.countplot(data=mpg, x='drv', order=['4', 'f', 'r'])
# plt.show()

# 빈도수 높은 순으로 정렬
sns.countplot(data=mpg, x='drv', order=mpg['drv'].value_counts().index)
plt.show()






















