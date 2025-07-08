# seoul_pubbike.py
# 서울시 공공자전거 따릉이 대여 현황 데이터
# 선형회귀 실습 : minutes(이용 시간<분단위>)로 distance 예측

# 라이브러리
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 로딩
df = pd.read_csv('assets/seoul_pubbike_2412.csv')
#print(df.head())
#df.info()

# 2. 필요 컬럼 추출
df = df[['minutes', 'distance']]

# 3. 이상치 제거 : 대여시간 0이거나 거리가 0인 경우 제거
df = df[(df['minutes'])>0 & (df['distance']>0)]

# 4. 데이터 분할
X = df[['minutes']]
y = df['distance']

# 5. 훈련/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 7. 예측
y_pred = model.predict(X_test)

# 8. 평가 지표
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'RMSE: {rmse:.2f}')
print(f'r2 score: {r2:.2f}')

# 9. 시각화
plt.rc("font", family='Malgun Gothic')
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='실제 데이터')
plt.plot(X_test, y_pred, color='red', label='회귀선')
plt.xlabel('대여 시간')
plt.ylabel('이동 거리')
plt.title('따릉이 대여시간/거리 선형회귀 예측 그래프')
plt.legend()
plt.grid(True)
plt.show()





























