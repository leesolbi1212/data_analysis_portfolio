# layer2_improved.py
# sigmoid 활성화 함수 기반 이진 분류 모델 (입력 정규화, 하이퍼파라미터 튜닝, EarlyStopping 적용)

from tensorflow.keras.models import Sequential               # 케라스 순차 모델 클래스
from tensorflow.keras.layers import Dense                      # 전결합(Dense) 레이어
from tensorflow.keras.optimizers import Adam                    # Adam 옵티마이저
from tensorflow.keras.callbacks import EarlyStopping           # 조기 종료 콜백
from sklearn.preprocessing import MinMaxScaler                 # 입력 피처 정규화
import numpy as np                                             # 수치 연산용

# 1) 데이터 준비
# X: [공부시간, 커피 수], y: [불합격=0, 합격=1]
X = np.array([[1, 0],
              [2, 1],
              [3, 1],
              [4, 2],
              [5, 3]])
y = np.array([0, 0, 0, 1, 1])

# 2) 입력 데이터 정규화 (0~1 범위로 스케일링)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # fit: min/max 계산, transform: 스케일링 적용

# 3) 모델 구성
model = Sequential()
model.add(Dense(8,                # 은닉층 뉴런 8개
                input_dim=2,      # 입력 특성 2차원 지정
                activation='relu' # ReLU 활성화로 비선형성 부여
               ))
model.add(Dense(1,                # 출력층 뉴런 1개 → 확률값 예측
                activation='sigmoid'  # Sigmoid 활성화로 0~1 확률 출력
               ))

# 4) 모델 컴파일: Adam 옵티마이저(학습률 조정), 이진분류 손실, 정확도 지표
optimizer = Adam(learning_rate=0.01)  # 기본 0.001 → 0.01로 조정해 빠른 수렴 시도
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5) EarlyStopping 콜백 설정: val_loss 기준 10회 연속 개선 없으면 학습 중단
es = EarlyStopping(monitor='val_loss',
                   patience=10,
                   restore_best_weights=True)

# 6) 모델 학습: 검증셋 20% 분리, 최대 500 epochs, EarlyStopping 적용
history = model.fit(X_scaled,       # 정규화된 입력
                    y,              # 레이블(0/1)
                    epochs=500,     # 최대 반복 횟수
                    validation_split=0.2,  # 20%는 검증용
                    callbacks=[es], # 조기 종료 콜백
                    verbose=1       # 학습 진행바 출력
                   )

# 7) 예측 테스트: 새로운 샘플도 동일한 스케일링 적용 후 예측
test_data = np.array([[3, 2],       # 공부 3h, 커피 2잔
                      [5, 1],       # 공부 5h, 커피 1잔
                      [1, 0]])      # 공부 1h, 커피 0잔
test_scaled = scaler.transform(test_data)  # 스케일 변환
predictions = model.predict(test_scaled)   # 확률 예측 (0~1)

# 8) 결과 출력: threshold=0.5로 분류, 소수점 4자리까지 표시
for i, prob in enumerate(predictions):
    label = '합격' if prob[0] >= 0.5 else '불합격'
    print(f"입력:{test_data[i]}, 예측확률:{prob[0]:.4f}, 분류: {label}")
