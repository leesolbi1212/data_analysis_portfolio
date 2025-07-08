# layer5.py
# 손실함수 Categorical Cross Entropy : 다중 클래스 분류용

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

# 입력 : [공부시간, 수면시간]
X = np.array([[1, 6], [2, 5], [3, 4], [4, 3], [5, 2]])

# 출력(등급) : 0 불합격, 1 보통, 2 합격
y = np.array([0, 0, 1, 2, 2])
y = to_categorical( # 원-핫 인코딩
    y,
    num_classes=3 # 등급의 수 (분류의 수)
)

# 모델
model = Sequential()
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax')) # 분류 수 만큼 출력

# 모델 컴파일
model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
model.fit(X, y, epochs=200, verbose=0)

# 예측 / 출력
pred = model.predict(np.array([[3, 3]]))
print(f'예측 결과 : {pred}')
print(f'예측 등급 : {np.argmax(pred)}') # 가장 확률 높은 등급 출력

















