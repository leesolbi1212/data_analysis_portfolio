# datagraph_box.py
# 박스 그래프
# 변수의 값의 범위를 표현

# 라이브러리 임포트
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터프레임
mpg = pd.read_csv('assets/mpg.csv')

# 박스 그래프
sns.boxplot(data=mpg, x='drv', y='hwy')
plt.show()





















