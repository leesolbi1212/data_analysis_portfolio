# recurrent_ex.py (완료)
# 순환신경망 실습

# 주제 : 기상청 데이터를 활용한 일일 평균기온 예측
# 데이터 : 기상자료개방포털에서 서울의 일최고기온/일평균기온 CSV파일 다운로드
# 예측목표 : 과거 30일간의 기온데이터를 이용해 다음 날의 기온을 예측
# 날짜, 평균기온 CSV파일을 사용

# 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential # 순차 모델 템플릿
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense  # RNN/LSTM/GRU 셀, 출력층(Dense)

# 1. 기상자료개방포털에 있는 서울_일평균기온.csv 로딩
df = pd.read_csv('assets/seoul_temp.csv', encoding='utf-8')
df['date'] = df['date'].astype(str).str.strip()            # 문자열 앞뒤 공백 제거
df = df[df['date'] != '']                                   # 빈 문자열 제거
df = df.dropna(subset=['date'])                             # 결측값 제거
df['date'] = pd.to_datetime(df['date'], errors='coerce')    # 에러가 나면 NaT로 처리
df = df.dropna(subset=['date'])                             # NaT 제거
df = df.sort_values('date') # 날짜 순으로 정렬

# 2. 기온 데이터 전처리
temp = df['avg_temp'].fillna(method='ffill').values  # 결측치 앞 값으로 채움

# 3. 시계열 입력/출력 만들기
def make_dataset(data, window=30):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window]) # 과거 window 일치 입력
        y.append(data[i+window]) # 바로 다음 날 출력
    return np.array(X), np.array(y)

X, y = make_dataset(temp, window=30)
X = X.reshape((X.shape[0], X.shape[1], 1)) # (샘플 수, 30, 1)

# 4. 훈련/테스트 나누기
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. 모델 빌딩 (구성) 함수
# 아무런 하이퍼파라미터 튜닝 없이 베이스라인 모델로 구성 (모델 성능 비교, 검증할 때 기준점으로 삼음)
# 튜닝 가능한 하이퍼파라미터 : 유닛수, 레이어 깊이, 드롭아웃/정규화, 학습률, 배치 사이즈/에포크 수, 할성화 함수 변경, 조기종료, 모델 체크포인트
def build_model(cell_type='RNN'):
    model = Sequential()
    if cell_type == 'RNN':
        model.add(SimpleRNN(32, input_shape=(30, 1))) # 셀 유닛수 32, 레이어 1개
    elif cell_type == 'LSTM':
        model.add(LSTM(32, input_shape=(30, 1)))
    elif cell_type == 'GRU':
        model.add(GRU(32, input_shape=(30, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# 6. 세 가지 모델 학습 및 예측
models, preds, rmses = {}, {}, {} #변수 3개를 빈 딕셔너리로 초기화
# 왜 딕셔너리? :각각의 모델·예측·RMSE 값을
# models['LSTM'], preds['GRU'], rmses['RNN'] 처럼
# 문자열 키로 바로 꺼내 쓰기 위해서.
# 리스트로 받으면 숫자 인덱스를 써야 해서 가독성이 떨어지기 때문
for cell in ['RNN', 'LSTM', 'GRU']:
    print(f"▶ {cell} 모델 학습 시작...")
    m = build_model(cell)
    m.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    pred = m.predict(X_test).flatten()
    models[cell] = m
    preds[cell] = pred
    rmses[cell] = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{cell} 모델 RMSE: {rmses[cell]:.3f}")

# 7. 결과 시각화 (x축을 날짜로 지정)
plt.rc("font", family="Malgun Gothic")
plt.figure(figsize=(14, 6))

# 테스트 구간에 해당하는 날짜 생성
test_dates = df['date'].values[-len(y_test):]

plt.plot(test_dates, y_test, label='실제 기온', color='black')  # x축에 날짜 사용
for cell in preds:
    plt.plot(test_dates, preds[cell], label=f'{cell} 예측')

plt.title('기상청 기온 예측: RNN vs LSTM vs GRU')
plt.xlabel('날짜')
plt.ylabel('평균기온 (℃)')
plt.xticks(rotation=45)  # 날짜 라벨 회전
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 8. RMSE 출력
print("\n 모델별 RMSE (Root Mean Squared Error)")
for cell in rmses:
    print(f"{cell}: {rmses[cell]:.3f}")
