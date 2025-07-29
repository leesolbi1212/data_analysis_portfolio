# rnn_imdb.py (완료)
# RNN 모델

# IMDB 데이터
# https://www.imdb.com/
# 영화, TV프로그램, 배우, 감독, 제작자 등 영상컨텐츠에 대한 데이터셋
'''
목적	: 감성 분석 (긍정 vs 부정 리뷰 분류)
제공처 : Stanford AI Lab, Keras 등
데이터 수 : 50,000개의 영화 리뷰 텍스트 (訓練: 25,000개 / 테스트: 25,000개)
레이블 : 0 = 부정, 1 = 긍정
형태	: .txt 파일 또는 파이썬 내장 API로 불러올 수 있음 (e.g., keras.datasets.imdb)
활용	: NLP 기초 튜토리얼, 텍스트 전처리, RNN/LSTM/GRU 모델 학습 등
'''

# 라이브러리
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# 데이터셋
(X_train, y_train), (X_test, y_test) = \
    imdb.load_data(num_words=10000) # 가장 많이 등장한 상위 10000개 단어 사용

# 시퀀스 길이 맞추기 (padding)
X_train = pad_sequences(X_train, maxlen=500) # 각 리뷰의 단어 시퀀스 길이를 500으로 맞춤
X_test = pad_sequences(X_test, maxlen=500)

# 모델 정의
model = Sequential([
    Embedding(10000, 32), # Embedding Layer : 정수 인덱스를 32차원 밀집 벡터로 변환
    SimpleRNN(32), # RNN(순환신경망) 계층, 시퀀스 데이터를 처리, 출력차원:32
    Dense(1, activation='sigmoid') # 이진 분류이므로 유닛 1개, sigmoid 활성화함수 사용
])

# 모델 컴파일
model.compile(
    optimizer='adam', # 최적화 함수
    loss='binary_crossentropy', # 손실 함수
    metrics=['accuracy'] # 성능 지표
)

# 모델 학습
model.fit(
    X_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

# 시각화
import matplotlib.pyplot as plt
history = model.history.history
plt.rc("font", family='Malgun Gothic')
plt.plot(history['accuracy'], label='train accuracy')
plt.plot(history['val_accuracy'], label='val accuracy')
plt.legend()
plt.title('정확도 변화 추이')
plt.show()
