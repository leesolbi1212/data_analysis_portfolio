# lstm_samsung.py
# LSTM모델을 이용한 삼성전자 주가 예측

# 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# 데이터셋
df = pd.read_csv('assets/samsung_stock.csv')
df['Date'] = pd.to_datetime(df['Unnamed: 0']) # 문자타입 > 날짜타입
df = df[['Date', 'Adj Close']].sort_values('Date') # 날짜, 조정종가 추출 및 정렬
df.set_index('Date', inplace=True) # Date를 인덱스로 설정

# 정규화 : 학습 안정성 향상을 위해
scaler = MinMaxScaler() # 기본적으로 0~1사이 값으로 정규화
scaled_data = scaler.fit_transform(df[['Adj Close']]) # 조정종가 정규화

# LSTM 학습용 시퀀스 데이터 생성 (60일)
sequence_len = 60 # 과거 60일치 데이터를 사용해서 예측
X, y = [], []
for i in range(sequence_len, len(scaled_data)):
    X.append(scaled_data[i-sequence_len:i, 0]) # 입력 시퀀스
    y.append(scaled_data[i, 0]) # 타겟 : 다음날의 종가
    
# 독립변수 / 종속변수 numpy 배열로 변환
X, y = np.array(X), np.array(y)

# LSTM 입력 형태로 reshape (samples, time steps, features)
X = X.reshape(
    X.shape[0], # 샘플 수 : 전체 데이터셋에서 학습에 사용할 입력 시퀀스의 개수
    X.shape[1], # 타임 스텝 : 하나의 시퀀스에 포함되는 과거 날짜 수
    1 # 피쳐 수 : 변수의 개수 (조정 종가)
)

# LSTM 모델 구성
model = Sequential()
model.add(
    LSTM(
        50, # 유닛 수
        return_sequences=False, # 마지막 시점 출력만 사용
        input_shape=(X.shape[1], 1) # 60일의 1차원 시계열 입력
    )
)
model.add(Dense(1)) # 예측값 1개

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 훈련
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# 예측을 위한 시퀀스 (마지막 60일)
last_sequence = scaled_data[-sequence_len:] # 마지막 60일의 정규화된 값
forecast_sequence = last_sequence.reshape(1, sequence_len, 1) # LSTM 입력형태로 reshape

# 미래 날짜 설정 및 예측 반복
last_date = df.index[-1] # 현재 데이터의 마지막 날짜
target_date = datetime(2026, 12, 31) # 예측 종료 날짜

future_dates = [] # 예측 날짜 리스트
future_predictions = [] # 예측 주가 리스트

# 반복적 1일씩 예측
while last_date < target_date:
    next_pred = model.predict(forecast_sequence)[0][0] # 다음날 예측값
    future_predictions.append(next_pred) # 예측결과 저장
    last_date += timedelta(days=1) # 날짜 1일 증가
    while last_date.weekday() >= 5: # 주말(토/일) 스킵
        last_date += timedelta(days=1)
    future_dates.append(last_date) # 평일 날짜만 추가
    next_seq = np.append(forecast_sequence[0, 1:, 0], next_pred) # 시퀀스 갱신
    forecast_sequence = next_seq.reshape(1, sequence_len, 1) # 다시 reshape

# 예측 결과를 역정규화
predicted_prices = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
)

# 결과 시각화
plt.rc("font", family='Malgun Gothic')
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Adj Close'], label='실제 주가')
plt.plot(future_dates, predicted_prices, label='LSTM 예측 주가', linestyle='--')
plt.title('LSTM모델로 예측한 2026년까지의 삼성전자 주가변동그래프')
plt.xlabel('날짜')
plt.ylabel('조정종가 (원)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()










































