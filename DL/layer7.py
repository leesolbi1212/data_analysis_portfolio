# layer7
# 최적화 함수

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import matplotlib.pyplot as plt

# 입력과 정답
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([50, 60, 70, 80, 90])

# 모델 생성 함수 (Optimizer만 다르게)
def build_model(optimizer):
    model = Sequential()
    model.add(Dense(8, input_dim=1, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mse')
    return model

# 결과 저장
results = {}

# Optimizer 리스트
optimizers = {
    'SGD': SGD(learning_rate=0.01),
    'RMSprop': RMSprop(learning_rate=0.01),
    'Adam': Adam(learning_rate=0.01),
}

# 학습 및 예측
for name, opt in optimizers.items():
    model = build_model(opt)
    history = model.fit(X, y, epochs=200, verbose=0)
    results[name] = history.history['loss']

# 손실 비교 그래프
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(8, 5))
for name, losses in results.items():
    plt.plot(losses, label=name)
plt.title("Optimizer별 손실 함수 감소 비교")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.show()