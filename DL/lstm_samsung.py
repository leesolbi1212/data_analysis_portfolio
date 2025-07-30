# lstm_samsung.py
# LSTM모델을 이용한 삼성전자 주가 예측

# 라이브러리
import pandas as pd # 데이터 프레임 생성 조작
import numpy as np # 수치 배열 연산
import matplotlib.pyplot as plt # 그래프 시각화
from sklearn.preprocessing import MinMaxScaler # 정규화
from tensorflow.keras.models import Sequential # 순차 모델 생성기
from tensorflow.keras.layers import LSTM, Dense # LSTM 셀, 출력층 레이어
from datetime import datetime, timedelta # 날짜 계산 유틸

# 데이터셋
df = pd.read_csv('assets/samsung_stock.csv') #csv 파일에서 모든 컬럼 로드
df['Date'] = pd.to_datetime(df['Unnamed: 0']) # 'Unnamed: 0' 열을 datetime 타입으로 변환
df = df[['Date', 'Adj Close']].sort_values('Date') # 날짜, 조정종가 추출 및 날짜 순 정렬
df.set_index('Date', inplace=True) # Date를 인덱스로 설정

# 정규화(0~1) : 학습 안정성 향상을 위해
scaler = MinMaxScaler() # 스케일러 객체 생성
scaled_data = scaler.fit_transform(df[['Adj Close']]) # 조정 종가 정규화

# LSTM 학습용 시퀀스 데이터 생성 (과거 60일 > 다음날 예측)
sequence_len = 60 # 입력 시퀀스 길이 과거 N일
X, y = [], [] # 빈 리스트로 초기화
for i in range(sequence_len, len(scaled_data)): # 시퀀스를 만들 수 있는 구간만큼 반복
    X.append(scaled_data[i-sequence_len:i, 0]) # 과거 60일치 종가를 x에 추가
    y.append(scaled_data[i, 0]) # 타겟 : 다음날의 종가, 61번째 날 종가를 y에 추가
    
# 독립변수 / 종속변수 numpy 배열로 변환
X, y = np.array(X), np.array(y)

# LSTM 입력 형태로 reshape (samples, time steps, features)
X = X.reshape(
    X.shape[0], # 샘플 수 : 전체 데이터셋에서 학습에 사용할 입력 시퀀스의 개수
    X.shape[1], # 타임 스텝 : 하나의 시퀀스에 포함되는 과거 날짜 수
    1 # 피쳐 수 : 변수의 개수 (조정 종가)
)

# LSTM 모델 구성
model = Sequential() # 순차 모델 시작
model.add(
    LSTM(
        50, # 유닛 수
        return_sequences=False, # 마지막 시점 출력만 사용
        input_shape=(X.shape[1], 1) # 60일의 1차원 시계열 입력 (60, )
    )
)
model.add(Dense(1)) # 예측값 1개, 회귀 예측값

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 훈련 (10회 학습)
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# 미래 예측을 위한 초기 시퀀스 준비 (마지막 60일)
last_sequence = scaled_data[-sequence_len:] # 마지막 60일의 정규화된 값
forecast_sequence = last_sequence.reshape(1, sequence_len, 1) # LSTM 입력형태로 reshape

# 미래 날짜와 예측값 저장 변수 초기화
last_date = df.index[-1] # 현재 데이터의 마지막 날짜
target_date = datetime(2027, 12, 31) # 예측 종료 날짜

future_dates = [] # 예측 날짜 리스트
future_predictions = [] # 예측 주가 리스트

# 1일씩 반복 예측 루프
while last_date < target_date:
    next_pred = model.predict(forecast_sequence)[0][0] # 다음날 예측값
    future_predictions.append(next_pred) # 예측 결과 리스트에 저장
    last_date += timedelta(days=1) # 날짜 1일 증가
    while last_date.weekday() >= 5: # 주말(토/일) 스킵
        last_date += timedelta(days=1)
    future_dates.append(last_date) # 평일 날짜만 추가
    next_seq = np.append(forecast_sequence[0, 1:, 0], next_pred) # 시퀀스 갱신 : 맨 앞값 제거하고 예측값 추가
    forecast_sequence = next_seq.reshape(1, sequence_len, 1) # 다시 reshape1q1`

# 예측 결과를 역정규화 -> 실제 가격 단위 (원)로 변환
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
