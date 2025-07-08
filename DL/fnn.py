# fnn.py

# FNN (Fully-Connected Neural Network : 완전 연결 신경망)
# 모든 신경망의 출력이 다음 신경망의 입력으로 사용되는 신경망

# 라이브러리
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from keras.datasets import mnist

# train / test 분리
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 시퀀셜 모델 생성
# 모델에 두 개의 Dense레이어(완전연결신경망) 추가
# 첫번째 Dense : 출력뉴런수(512), 입력크기(784), 출력크기(512)
# 두번째 Dense : 출력뉴런수(10), 입력크기(512), 출력크기(10)
network = Sequential()
network.add(
    layers.Dense(
        512, # 출력 뉴런 수
        activation='relu', # 활성화 함수
        input_shape=(28*28,) # 입력벡터 28*28=784차원, 이미지 크기
    )
)
network.add(
    layers.Dense(
        10, # 출력 뉴런 수, 최종 결과는 10개의 클래스 중 하나
        activation='softmax' # 활성화 함수
    )
)

# 컴파일
# rmsprop
# 진동이 심한 경사면에서는 학습률을 줄이고 완만한 경사면에서는 학습률을 유지해서
# 결과가 안정적일 수 있도록 유도
# categorical_crossentropy
# 다중 클래스 분류에 사용되며 레이블이 원-핫 인코딩 되어 있어야 함
network.compile(
    optimizer = 'rmsprop', # 최적화 함수
    loss = 'categorical_crossentropy', # 손실 함수, 다중 클래스 분류
    metrics = ['accuracy'] # 지표:정확도
)

# 차원 변경
train_images = train_images.reshape((60000, 28*28))
test_images = test_images.reshape((10000, 28*28))

# 0~1 값으로 스케일링
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# 범주 분리
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#print(train_labels)
#print(test_labels)

# 훈련
history = network.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)

## 시각화
import matplotlib.pyplot as plt
plt.rc("font", family="Malgun Gothic")

# 훈련 정확도/손실 시각화
# plt.figure(figsize=(12, 4))
# plt.plot(history.history['accuracy'], label='훈련 정확도')
# plt.plot(history.history['val_accuracy'], label='검증 정확도')
# plt.plot(history.history['loss'], label='훈련 손실')
# plt.plot(history.history['val_loss'], label='검증 손실')
# plt.title('훈련/검증 정확도/손실')
# plt.xlabel('Epoch 수')
# plt.ylabel('정확도')
# plt.legend()
# plt.show()

# 테스트 이미지 일부와 예측 결과 시각화
# import numpy as np
# predictions = network.predict(test_images)
# plt.figure(figsize=(15, 5))
# for i in range(10):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(test_images[i].reshape(28,28), cmap='gray')
#     predicted_label = np.argmax(predictions[i]) # 예측 레이블
#     true_label = np.argmax(test_labels[i]) # 실제 레이블
#     color = 'green' if predicted_label==true_label else 'red'
#     plt.title(f'예측: {predicted_label}\n실제: {true_label}', color=color)
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

# 혼동 행렬 (Confusion Matrix)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# 예측 > 정수형 라벨로 변환
predictions = network.predict(test_images)
y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(test_labels, axis=1)

# 혼동 행렬 생성
conf_mat = confusion_matrix(y_true, y_pred)

# 혼동행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title("혼동행렬 (Confusion Matrix)")
plt.xlabel("예측 레이블")
plt.ylabel("실제 레이블")
plt.show()




















