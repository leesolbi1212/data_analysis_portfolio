# layer3_improved.py (완료)
# 회귀 예제에 “검증 분리”, “EarlyStopping”, “MAE 지표” 추가

from tensorflow.keras.models import Sequential    # 케라스 순차 모델
from tensorflow.keras.layers import Dense         # 전결합(Dense) 레이어
from tensorflow.keras.callbacks import EarlyStopping  # 조기 종료 콜백
import numpy as np                                 # 수치 연산

# 1) 데이터 준비
X_raw = np.array([[1], [2], [3], [4], [5]])        # 공부시간
y_raw = np.array([50, 60, 70, 80, 90])             # 시험점수

# 2) Z‑score 정규화 함수
def normalize(data):
    return (data - np.mean(data)) / np.std(data)

# 3) 정규화 적용
X = normalize(X_raw)
y = normalize(y_raw)

# 4) 모델 구성
model = Sequential()
model.add(
    Dense(
        8,                 # 은닉 뉴런 8개
        input_dim=1,       # 입력 특성 1차원
        activation='relu'  # 비선형성 도입
    )
)
model.add(
    Dense(
        1                  # 선형 노드 1개 → 회귀 예측
        # activation='linear'  # 기본값이 선형이므로 생략 가능
    )
)

# 5) 모델 컴파일
#    - loss: MSE (평균제곱오차)
#    - metrics: MAE (평균절대오차) 추가 → 오차 크기 직관적으로 확인
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# 6) EarlyStopping 설정
#    - val_loss(검증 손실) 기준으로 10회 연속 개선 없으면 중단
#    - restore_best_weights=True: 최적 검증 모델 가중치 복원
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 7) 모델 학습
#    - validation_split=0.2: 전체의 20%를 검증용으로 자동 분리
#    - callbacks=[es]: 조기 종료 적용
#    - verbose=1: epoch별 진행바 및 loss/mae 출력
history = model.fit(
    X, y,
    epochs=200,
    validation_split=0.2,
    callbacks=[es],
    verbose=1
)

# 8) 예측
#    - 입력 3시간도 정규화 → 예측 → 역정규화
test = np.array([[3]])
test_norm = normalize(test)
pred_norm = model.predict(test_norm)
pred = pred_norm * np.std(y_raw) + np.mean(y_raw)

print(f"\n공부 3시간 > 예측 점수 {pred[0][0]:.2f}")

# 9) 학습 결과 요약 (MAE 지표 포함)
final_train_mae = history.history['mae'][-1]
final_val_mae   = history.history['val_mae'][-1]
print(f"최종 Train MAE: {final_train_mae:.4f}")
print(f"최종 Val   MAE: {final_val_mae:.4f}")
