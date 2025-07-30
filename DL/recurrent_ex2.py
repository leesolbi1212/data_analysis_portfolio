# (추가 학습 해보기)
import pandas as pd                                 # CSV 로딩·전처리
import numpy as np                                  # 수치 계산
import matplotlib.pyplot as plt                     # 시각화
from sklearn.preprocessing import MinMaxScaler       # 정규화 도구
from sklearn.metrics import mean_squared_error      # 성능 평가 (RMSE 계산)
from tensorflow.keras.models import Sequential      # 순차 모델 생성
from tensorflow.keras.layers import LSTM, Dropout, Dense  # LSTM, Dropout, 출력층
from tensorflow.keras.callbacks import EarlyStopping    # 과적합 방지
# ───────────────────────────────────────────────────────────────────────────────
# 2. 데이터 로드 및 정제
df = pd.read_csv('assets/seoul_temp.csv', encoding='utf-8')      # 원본 CSV 로딩
df['date'] = df['date'].astype(str).str.strip()                 # 문자열 공백 제거
df['date'] = pd.to_datetime(df['date'], errors='coerce')       # 날짜 파싱, 실패 시 NaT
df = df.dropna(subset=['date', 'avg_temp'])                     # 날짜/기온 결측 제거
df = df.sort_values('date').reset_index(drop=True)              # 날짜 오름차순 정렬

# 3. 계절성(연간 주기) 인코딩: dayofyear → sin, cos
df['dayofyear'] = df['date'].dt.dayofyear                       # 1~365
df['sin_doy']   = np.sin(2 * np.pi * df['dayofyear'] / 365)     # 사인 변환
df['cos_doy']   = np.cos(2 * np.pi * df['dayofyear'] / 365)     # 코사인 변환

# 4. 결측 기온 보간 및 정규화
df['avg_temp'] = df['avg_temp'].interpolate()                   # 선형 보간
scaler = MinMaxScaler()                                         # 0~1 스케일러
df['temp_scaled'] = scaler.fit_transform(df[['avg_temp']])      # 스케일링

# 5. 슬라이딩 윈도우 데이터셋 생성 (윈도우=365일, 피처=3개)
window = 365                                                    # 1년치 히스토리
features = ['temp_scaled', 'sin_doy', 'cos_doy']                # 사용할 피처
data = df[features].values                                      # (N, 3) 배열로 추출
X, y = [], []
for i in range(window, len(data)):
    X.append(data[i-window : i])     # 과거 365일치 (365×3)
    y.append(data[i, 0])             # 타깃: 당일 temp_scaled
X = np.array(X)  # (샘플 수, 365, 3)
y = np.array(y)  # (샘플 수,)

# 6. 훈련/테스트 분리 (80% / 20%)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 7. 모델 구성 (다중 LSTM + Dropout + Huber Loss + EarlyStopping)
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(window, len(features))))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))                                        # 스케일된 기온 1개 예측
model.compile(optimizer='adam', loss='huber')              # Huber Loss 사용

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,      # 10회 연속 개선이 없으면 중단
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 8. 예측 및 역정규화
y_pred_scaled = model.predict(X_test).flatten()            # 스케일된 예측값
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
y_true = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

# 9. 평가 및 시각화
rmse = np.sqrt(mean_squared_error(y_true, y_pred))         # RMSE 계산
print(f"개선 모델 RMSE: {rmse:.3f} ℃")                      # 출력

plt.figure(figsize=(12,5))
test_dates = df['date'].values[-len(y_true):]              # 테스트 구간 날짜
plt.plot(test_dates, y_true, label='실제 기온', color='black')
plt.plot(test_dates, y_pred, label='개선 모델 예측', color='tab:orange')
plt.title('계절성 인코딩 + 긴 윈도우 LSTM 예측 결과')
plt.xlabel('날짜'); plt.ylabel('평균기온 (℃)')
plt.xticks(rotation=45); plt.legend(); plt.grid(); plt.tight_layout()
plt.show()

