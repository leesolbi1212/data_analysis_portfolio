# daily_temp.py
# 실습 : 기상청 기온 데이터로 이상기온 탐지를 위한 데이터 검색
#        서울지역의 날짜, 최고기온, 최저기온 데이터 검색
#       과거 온도 데이터를 기반으로 이상기온을 탐지
#       IsolationForest : 이상치 탐지를 위한 머신러닝 기법, 비지도 학습

# 라이브러리
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. 데이터 불러오기
df = pd.read_csv('assets/daily_temp_seoul.csv') # 날짜, 최고기온, 최저기온 포함
df['date'] = pd.to_datetime(df['date']) # 문자데이터를 날짜데이터로 변경
df['mean_temp'] = (df['high_temp'] + df['low_temp']) / 2 # 날짜별 평균기온

# 2. 모델 학습
X = df[['mean_temp']]
iso_forest = IsolationForest(contamination=0.02) # 상위 2%는 이상치로 간주
df['outlier'] = iso_forest.fit_predict(X) # 이상치 예측

# 3. 시각화
plt.rc("font", family='Malgun Gothic')
plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['mean_temp'], label='평균 기온')
plt.scatter(
    df[df['outlier']==-1]['date'],
    df[df['outlier']==-1]['mean_temp'],
    color='red',
    label='이상치'
)
plt.legend()
plt.title('서울 이상기온 탐지')
plt.show()
























