# layer.py
# 입력층, 은닉층, 출력층
from jinja2.optimizer import optimize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 입력 : 공부시간과 수면시간, 출력 : 시험점수
X = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
y = np.array([50, 60, 70, 80, 90]).reshape(-1, 1) # 2차원 배열로 변환

# 입력/출력 데이터 정규화
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# 모델 구성

# 각 층들을 순서적으로 연결하는 시퀀셜 모델
model = Sequential()

# 모델에 입력층을 추가
model.add(
    Input(
        shape = (2,)
    )
)

# 모델에 은닉층을 추가
model.add(
    Dense(
        10, # 데이터 수
        input_dim=2, # 2차원
        activation='relu' # 활성화 함수
    )
)

# 모델에 출력층을 추가
model.add(
    Dense(1) # 데이터 수
)

# 모델 컴파일
model.compile(
    optimizer = 'adam', # 최적화 함수
    loss = 'mse' # 손실 함수
)

# 모델 학습
model.fit(
    X_scaled, # 정규화한 입력데이터
    y_scaled, # 정규화한 출력데이터
    epochs=100, # 학습 반복 회수
    verbose=0 # 학습 상태에 대한 출력 설정 0:출력 없음, 1:진행바형태 출력, 2: epoch별 출력
)

# 예측 결과 (스케일 복원 포함)
y_pred_scaled = model.predict(
    X_scaler.transform(np.array([[3, 2]])) # 공부 3시간, 수면 2시간
)
y_pred = y_scaler.inverse_transform(y_pred_scaled) # 점수 예측값

print("예측 시험 점수:", y_pred[0][0])













































