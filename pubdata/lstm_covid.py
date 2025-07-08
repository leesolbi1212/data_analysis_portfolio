# lstm_covid.py
# 공공데이터 (질병관리청 코로나19데이터) 활용한 LSTM 모델 활용 실습
# 일자별 코로나 확진자 수 데이터를 기반으로 향후 며칠간 확진자 수를 예측
# 데이터 출처 : 질병관리청 코로나19 데이터
#              https://dportal.kdca.go.kr/pot/cv/trend/dmstc/selectMntrgSttus.do

# 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. 데이터 불러오기
df = pd.read_csv('assets/covid.csv')  # 컬럼: date, confirmed
df['date'] = pd.to_datetime(df['date']) # 문자열 > 날짜
df['confirmed'] = df['confirmed'].str.replace(',', '')
df['confirmed'] = pd.to_numeric(df['confirmed']) # 문자열 > 숫자
df = df[['date', 'confirmed']].set_index('date') # 인덱스를 date로
# print(df.head())
# df.info()

# 2. 정규화
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# 3. 시계열 데이터 생성 (최근 7일 데이터로 다음날 예측)
def create_sequence(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

sequence_length = 7
X, y = create_sequence(scaled_data, sequence_length)

# 4. LSTM 모델 설계
model = Sequential()
model.add(LSTM(
    units=50, # LSTM 셀의 출력 노드 수
    return_sequences=False, # 다음 LSTM레이어로 시퀀스를 넘기지 않음 (Dense로 직접 넘김)
    # 입력시퀀스의 형태, sequence_length:타임스텝 수, 1: 각 시점의 특성 수 
    input_shape=(sequence_length, 1)
))
model.add(Dense(1)) # 출력 뉴런 수 1 : 예측값이 1개

# 5. 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. 학습
model.fit(X, y, epochs=20, batch_size=16)

# 7. 예측
predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y.reshape(-1, 1))

# 8. 시각화
plt.rc("font", family="Malgun Gothic")
plt.figure(figsize=(12,6))

# 첫번째 7일은 제외
date_range = df.index[sequence_length:]
plt.plot(date_range, actual, label='Actual Confirmed')
plt.plot(date_range, predicted, label='Predicted Confirmed')
plt.xlabel('날짜')
plt.ylabel('확진자수')
plt.xticks(rotation=45)
plt.legend()
plt.title("코로나19 확진자 수 예측")
plt.show()


























