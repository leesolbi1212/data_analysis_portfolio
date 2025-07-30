# recurrent.py
# RNN, LSTM, GRU 비교

# 라이브러리
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense
import matplotlib.pyplot as plt

# 데이터셋
(X_train, y_train), (X_test, y_test) = \
    imdb.load_data(num_words=10000) # 가장 많이 등장한 상위 10000개 단어 사용

# 시퀀스 길이 맞추기 (padding)
X_train = pad_sequences(X_train, maxlen=500) # 각 리뷰의 단어 시퀀스 길이를 500으로 맞춤
X_test = pad_sequences(X_test, maxlen=500)

# 모델 구성 함수
def build_model(cell_type):
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=32, input_length=100))
    if cell_type == 'RNN':
        model.add(SimpleRNN(32))
    elif cell_type == 'LSTM':
        model.add(LSTM(32))
    elif cell_type == 'GRU':
        model.add(GRU(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# 모델 학습 및 결과 저장
results = {}
for cell in ['RNN', 'LSTM', 'GRU']:
    print(f'{cell} 모델 학습 중...')
    model = build_model(cell)
    history = model.fit(
        X_train,
        y_train,
        epochs = 10,
        batch_size = 64,
        validation_split = 0.2
    )
    results[cell] = history.history

# 결과 시각화
plt.rc("font", family="Malgun Gothic")
plt.figure(figsize=(12, 5))
for cell in results:
    plt.plot(results[cell]['val_accuracy'], label=f'{cell} 검증 정확도')
plt.title('RNN, LSTM, GRU 정확도 비교')
plt.xlabel('Epoch')
plt.ylabel('검증 정확도')
plt.legend()
plt.grid()
plt.show()
