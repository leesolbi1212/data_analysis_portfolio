# layer2.py
# sigmoid 활성화 함수

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 입력 데이터 (공부시간, 커피 수)
X = np.array([[1, 0], [2, 1], [3, 1], [4, 2], [5, 3]])

# 출력 데이터 (합격=1, 불합격=0)
y = np.array([0, 0, 0, 1, 1])

# 모델
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu')) # 은닉층
model.add(Dense(1, activation='sigmoid')) # 출력층

# 모델 컴파일
model.compile(
    optimizer='adam', # 최적화 함수 
    loss='binary_crossentropy', # 손실 함수
    metrics=['accuracy'] # 지표 : 정확도
)

# 모델 학습
model.fit(X, y, epochs=200, verbose=0)

# 예측 테스트
test_data = np.array([[3, 2], [5, 1], [1, 0]])
predictions = model.predict(test_data)

# 결과 출력
for i, pred in enumerate(predictions):
    print(f"입력:{test_data[i]}, 예측확률:{pred[0]:.4f}, \
        분류: {'합격' if pred[0]>=0.5 else '불합격'}")

























