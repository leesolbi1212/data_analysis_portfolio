# cnn_cat&dog.py
# 실습 : 고양이와 개 CNN 분류

# 개요
'''
데이터셋 : Tensorflow 개와 고양이 이미지 25,000장
분류 수 : 2개 클래스 (0:고양이, 1:개)
입력 크기 : 가로 180, 세로 180, 3원색 RGB
모델 구조 : Conv2D + MaxPooling + Dense
결과 : 정확도와 손실 시각화
'''

# 라이브러리
import matplotlib.pyplot as plt # 시각화
import numpy as np # 수치 연산
import PIL # 이미지 처리
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
import os

# 1. 데이터 다운로드 및 경로 설정
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_path = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=dataset_url, extract=True)
base_dir = os.path.join(os.path.dirname(zip_path), 'cats_and_dogs_filtered_extracted/cats_and_dogs_filtered')

# 훈련 데이터셋, 검증 데이터셋 경로
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# 2. 훈련/검증 데이터셋 생성
BATCH_SIZE = 32
IMG_SIZE = (180, 180)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names  # ['cats', 'dogs']

# 3. 데이터 시각화 (9개 이미지)
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(10, 6))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.suptitle("개와 고양이 이미지 예시", fontsize=16)
plt.tight_layout()
plt.show()

# 4. 데이터 정규화
# 데이터 정규화
normalization_layer = layers.Rescaling(1./255) # 0~1 범위로 정규화
# 훈련데이터 각각에 정규화 실시
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# 5. CNN 모델 생성 (Sequential 모델)
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),  # 0~1 정규화

    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)  # 출력층: 이진분류 → 1개 노드 (0: 고양이, 1: 개)
])

# 6. 모델 컴파일
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 7. 모델 학습 (epoch 10)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# 8. 학습결과 시각화 (훈련/검증 정확도, 훈련/검증 손실)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(acc, label='훈련 정확도')
plt.plot(val_acc, label='검증 정확도')
plt.title('정확도 변화')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='훈련 손실')
plt.plot(val_loss, label='검증 손실')
plt.title('손실 변화')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


























