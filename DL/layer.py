# layer.py (완료)

# 1. 필요한 라이브러리 임포트
from tensorflow.keras.models import Sequential #층을 순서대로 쌓는 케라스 모델 클래스
from tensorflow.keras.layers import Input, Dense #input : 입력층 정의, Dense : 완전 연결층
from sklearn.preprocessing import MinMaxScaler # 데이터를 0~1 구간으로 선형 이동, 확대/축소하는 전처리 도구
import numpy as np # 수치 연산용 배열 (array) 및 연산 함수 제공

# 2. 데이터 준비
# 입력 : 공부시간과 수면시간, 출력 : 시험점수
X = np.array([[1, 5], [2, 4], [3, 3], [4, 2], [5, 1]])
y = np.array([50, 60, 70, 80, 90]).reshape(-1, 1) # 2차원 배열로 변환 (5,) -> (5,1)

# 3. 입력/출력 데이터 정규화 : 학습 속도 및 안정성 향상
# 서로 다른 스케일의 피처(공부시간, 수면시간, 점수)를 동일 기준으로 맞춤
X_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
# 학습 데이터를 보고 min, max 계산 -> 각 값에 (value-min) / (max-min) 적용
X_scaled = X_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# 4. 모델 구성

# 각 층들을 순서적으로 연결하는 빈 시퀀셜 모델 생성
model = Sequential()

# 4-1. 모델에 입력층을 추가 : 입력 데이터 형태는 특성 2개
model.add(
    Input(
        shape = (2,)
    )
)

# 4-2. 모델에 은닉층을 추가
model.add(
    Dense(
        10, # 데이터 수(뉴런)
        input_dim=2, # 2차원 (Input(shape=(2,))와 중복이므로 Input만 남기고 지워도 됨)
        activation='relu' # 활성화 함수 : 0이하는 0 출력, 0초과입력은 입력값 그대로 출력, 비선형성 도입, 학습속도 빠름
    )
)

# 4-3. 모델에 출력층을 추가
model.add(
    Dense(1) # 데이터 수
)

# Q. 똑같이 Dense()로만 추가하는데, 왜 하나는 은닉층이고 하나는 출력층인지 어떻게 알 수 있을까?
# Dense 층의 역할은 추가되는 순서와 모델 설계자의 목적에 따라 결정됨.


# 5. 모델 컴파일
model.compile(
    optimizer = 'adam', # 최적화 함수 : 학습률을 자동 조정하는 확률적 경사하강법 변형
    loss = 'mse' # 손실 함수 : 회귀용 손실 함수, (실제값 - 예측값)^2의 평균
)

# 6. 모델 학습
model.fit(
    X_scaled, # 정규화한 입력데이터
    y_scaled, # 정규화한 출력데이터
    epochs=100, # 학습 반복 횟수
    verbose=0 # 학습 상태에 대한 출력 설정 0:출력 없음, 1:진행바형태 출력, 2: epoch별 출력
)

# 7. 예측 결과 (스케일 복원 포함)
y_pred_scaled = model.predict(
    X_scaler.transform(np.array([[3, 2]])) # 공부 3시간, 수면 2시간 새 입력 (0~1로 변환)
)
y_pred = y_scaler.inverse_transform(y_pred_scaled) # 점수 예측값 (inverse로 원래 점수 범위로 복원)

print("예측 시험 점수:", y_pred[0][0])


