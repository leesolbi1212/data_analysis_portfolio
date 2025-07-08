# layer3.py
# 손실함수 - Binary Cross Entropy (이진 교차 엔트로피) : 이진 분류용

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 공부시간(입력), 시험점수(출력)
X = np.array([[1], [2], [3], [4], [5]])
y_raw = np.array([50, 60, 70, 80, 90])

# 정규화 함수
# 데이터에서 데이터들의평균을 뺀값을 표준편차로 나눔
def normalize(data):
    return (data-np.mean(data)) / np.std(data)

# 정규화
X = normalize(X)
y = normalize(y_raw)

# 모델
model = Sequential()
model.add(Dense(8, input_dim=1, activation='relu'))
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy')

# 모델 학습
model.fit(X, y, epochs=200, verbose=0)

# 예측 / 출력
pred = model.predict(normalize(np.array([[3]])))
pred = pred * np.std(y_raw) + np.mean(y_raw) # 역정규화
print(f"공부 3시간 > 예측 점수 {pred[0][0]:.2f}")









