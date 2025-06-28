# datagraph_line.py
# 라인 그래프
# 시계열 데이터(시간의 흐름에 따라 변화하는 데이터) 표현

# 라이브러리 임포트
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc # 폰트 세팅을 위한 모듈 추가
font_path = "C:/Windows/Fonts/malgun.ttf" # 사용할 폰트명 경로 삽입
font = font_manager.FontProperties(fname = font_path).get_name()
rc('font', family = font)

# 데이터프레임
economics = pd.read_csv('C:/AI_2504/dataanalysis/assets/economics.csv')
economics.info()

# 라인 그래프
#sns.lineplot(data=economics, x='date', y='unemploy')
#plt.show()

# x축에 연도 표시
economics['date2'] = pd.to_datetime(economics['date']) # 날짜타입으로 변환
economics['year'] = economics['date2'].dt.year # 연도

# sns.lineplot(data=economics, x='year', y='unemploy') #x축, y축 설정
# plt.title ('연도별 실업자 수 추이')
# plt.show()

# 신뢰구간 표시 제거
# ci=None : 신뢰구간 표시 제거 (불필요한 음영 있음)
sns.lineplot(data=economics, x='year', y='unemploy', ci=None)
plt.title('연도별 실업자 수 추이')
plt.show()































