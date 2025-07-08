# ex-linearregression.py
# 선형회귀 실습
'''
실습 주제 : 선형회귀를 이용한 광고비 기반 매출 예측
실습 설명 : 회사는 다양한 광고 채널(예: TV, 라디오, 신문)에 광고비를 집행하고 있습니다.
          이때 광고비 지출이 매출(판매량)에 어떤 영향을 주는지 분석하고,
          광고비만 보고 매출을 예측할 수 있는 모델을 만들어봅니다.
          "모델 평가지표는 MSE, R^2으로 평가하고 시각화 해보기"
데이터셋 : Kaggle (https://www.kaggle.com/datasets/ashydv/advertising-dataset)
                 (advertising.csv)
주요 컬럼:
- TV : TV 광고비
- Radio : 라디오 광고비
- Newspaper : 신문 광고비
- Sales: 실제 판매량
'''

# 1. 필요한 라이브러리 불러오기
import pandas as pd                                # 데이터프레임 처리
import numpy as np                                 # 수치 계산
import matplotlib.pyplot as plt                    # 시각화 도구
from sklearn.linear_model import LinearRegression  # 선형 회귀 모델
from sklearn.model_selection import train_test_split  # 데이터 분할
from sklearn.metrics import mean_squared_error, r2_score  # 평가 지표

from ml.classification import X_train

# 2. 데이터 로딩
df = pd.read_csv('assets/advertising.csv')
#print(df.head())
#df.info()

# 3. 독립변수 / 종속변수 분리
X = df[['TV', 'Radio', 'Newspaper']] # 광고비
y = df['Sales'] # 판매량(매출)

# 4. 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5. 선형 회귀 모델 생성
model = LinearRegression()

# 6. 모델 학습
model.fit(X_train, y_train)

# 7. 모델 평가 지표 출력
# 회귀계수 : [0.05450927 0.10094536 0.00433665]
# TV 광고비를 1늘리면 판매량이 평균 0.05 증가
# Radio 광고비를 1늘리면 판매량이 평균 0.1 증가
# Newspaper 광고비를 1늘리면 판매량이 평균 0.004 증가
print('회귀계수(각 광고비의 영향) : ', model.coef_)
# 절편 : 4.714126402214127
# 광고비를 지출하지 않아도 측정되는 기본 판매량
print('절편(기본 판매량) : ', model.intercept_)

# 8. 예측
y_pred = model.predict(X_test)

# 9. 평가 지표 출력

# mse : 2.91
# 예측값과 실제값 간의 차이의 제곱 평균
# => 오차가 평균 2.91의 제곱만큼 발생
mse = mean_squared_error(y_test, y_pred)
print(f'평균제곱오차(MSE) : {mse:.2f}')

# r2 : 0.91
# 전체 판매량 변동 중에서 91%는 광고비로 설명 가능함
# 1에 가까울수록 좋음
r2 = r2_score(y_test, y_pred)
print(f'결정계수(R^2) : {r2:.2f}')

# 시각화
# 예측 결과 시각화
plt.figure(figsize=(8, 6))
plt.rc('font', family='Malgun Gothic')
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--')
plt.xlabel('실제 판매량')
plt.ylabel('예측 판매량')
plt.title('선형 회귀 예측 결과')
plt.grid(True)
plt.show()
















