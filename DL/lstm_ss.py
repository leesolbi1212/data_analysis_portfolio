# lstm_samsung_evaluation.py
# - 삼성전자 주가 시계열을 LSTM으로 예측하고
#   1) 단순 train/test 분할
#   2) walk-forward validation
#   3) sklearn TimeSeriesSplit
#   세 가지 방식으로 평가해 봅니다.

import pandas as pd                    # 데이터프레임
import numpy as np                     # 수치 계산
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import random

#— 1. 시드 고정 및 세션 초기화 —#
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.backend.clear_session()

#— 2. 데이터 로딩 & 전처리 함수 정의 —#
def load_preprocess(path, seq_len=60):
    # 1) CSV 로드 및 날짜 인덱스 설정
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Unnamed: 0'])
    df = df[['Date', 'Adj Close']].sort_values('Date').set_index('Date')
    # 2) MinMax 정규화
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Adj Close']])
    # 3) 시퀀스 생성
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X)
    y = np.array(y)
    # 4) LSTM 입력 형태로 reshape
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler, df.index[seq_len:]

#— 3. LSTM 모델 생성 함수 —#
def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=input_shape),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

#— 메인 실행 —#
if __name__ == "__main__":
    # 데이터 불러오기
    X, y, scaler, dates = load_preprocess('assets/samsung_stock.csv')

    # 1) 단순 Train/Test 분할 (80/20)
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model1 = build_lstm((X.shape[1], 1))
    model1.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    pred1 = model1.predict(X_test)
    mse1 = mean_squared_error(y_test, pred1)
    print(f"[단순분할] Test MSE: {mse1:.5f}")

    # 2) Walk-Forward Validation
    #    – 매 시점마다 모델 재학습 → 다음 시점 예측
    errors_wf = []
    for i in range(split, len(X)):
        # i는 예측하려는 시퀀스 인덱스
        Xi_train, yi_train = X[:i], y[:i]
        Xi_test, yi_test = X[i:i+1], y[i:i+1]
        model_wf = build_lstm((X.shape[1], 1))
        # 학습 에폭을 적게 잡아 속도 조절 가능 (여기선 5)
        model_wf.fit(Xi_train, yi_train, epochs=5, batch_size=32, verbose=0)
        pred = model_wf.predict(Xi_test)
        errors_wf.append((yi_test[0] - pred[0,0])**2)
    mse2 = np.mean(errors_wf)
    print(f"[Walk-Forward] Avg MSE: {mse2:.5f}")

    # 3) TimeSeriesSplit (n_splits=5)
    tscv = TimeSeriesSplit(n_splits=5)
    errors_ts = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        Xi_train, Xi_test = X[train_idx], X[test_idx]
        yi_train, yi_test = y[train_idx], y[test_idx]
        model_ts = build_lstm((X.shape[1], 1))
        model_ts.fit(Xi_train, yi_train, epochs=10, batch_size=32, verbose=0)
        pred = model_ts.predict(Xi_test)
        mse_fold = mean_squared_error(yi_test, pred)
        errors_ts.append(mse_fold)
        print(f"[TSplit Fold {fold+1}] MSE: {mse_fold:.5f}")
    mse3 = np.mean(errors_ts)
    print(f"[TimeSeriesSplit] Avg MSE: {mse3:.5f}")

    #— 최종 비교 —#
    print("\n=== 모델별 성능 비교 ===")
    print(f"단순분할       : {mse1:.5f}")
    print(f"Walk-Forward   : {mse2:.5f}")
    print(f"TimeSeriesSplit: {mse3:.5f}")
