# regression-linearregression.py
# 선형회귀

# 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # 선형회귀
from sklearn.datasets import load_diabetes # 당뇨병 데이터셋
from sklearn.model_selection import train_test_split # 훈련/테스트 데이터셋 분할

# 데이터셋 로딩
diabetes = load_diabetes()
#print(diabetes)

# 독립변수(X)와 종속변수(y) 분리
X = diabetes.data # 10개의 변수(특성, feature)을 가진 입력 데이터
y = diabetes.target # 예측 대상(당뇨병 진행률)

# 특성 선택
# : => X의 모든 행
# np.newaxis : 차원(축)을 하나 늘림
# 2 => 2번 인덱스 열을 가져옴
X = X[:, np.newaxis, 2] # 2차원 배열 형태로 선택 ([[], [], []])
#print(X)

# 훈련/테스트 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(X_train, y_train)

# 모델 정보 출력
#print(f'기울기(slope) : {model.coef_[0]:.2f}') # 기울기 소수점 2째자리까지
#print(f'절편(intercept) : {model.intercept_:.2f}') # 절편 소수점 2째자리까지

# 예측
y_pred = model.predict(X_test)

# 예측 결과 시각화
plt.rc('font', family='Malgun Gothic')
plt.scatter(X_test, y_test, color='blue', label='실제 데이터')
plt.plot(X_test, y_pred, color='red', label='예측(회귀) 선')
plt.xlabel('BMI(체질량 지수)')
plt.ylabel('당뇨병 진행률')
plt.title('선형 회귀 - 당뇨병 예측 (BMI)')
plt.legend()
plt.grid(True)
plt.show()



















































