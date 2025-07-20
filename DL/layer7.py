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
def build_model(optimizer): #입력 파라미터는 optimizer
    model = Sequential()
    model.add(Dense(8, input_dim=1, activation='relu')) # 뉴런 8개, ReLU
    model.add(Dense(1)) # 출력층 : 선형노드 1개, 회귀예측
    model.compile(optimizer=optimizer, loss='mse') # 지정된 옵티마이저 + MSE 손실
    return model

# 결과 저장
results = {}

# Optimizer 리스트
# 학습률을 모두 0.01로 동일 설정, 이름과 인스턴스를 매핑해 반복 처리 준비
optimizers = {
    'SGD': SGD(learning_rate=0.01),
    'RMSprop': RMSprop(learning_rate=0.01),
    'Adam': Adam(learning_rate=0.01),
}

# 학습 및 예측
for name, opt in optimizers.items():
    model = build_model(opt) # 모델 생성
    history = model.fit(X, y, epochs=200, verbose=0) # 200회 학습
    results[name] = history.history['loss'] #Epoch별 손실값 리스트 획득

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