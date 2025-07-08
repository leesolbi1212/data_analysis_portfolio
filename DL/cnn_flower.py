# cnn_flower.py
# CNN을 활용한 꽃이미지 분류

# 라이브러리
import matplotlib.pyplot as plt # 시각화
import numpy as np # 수치 연산
import PIL # 이미지 처리
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# 데이터 : 3670장의 꽃 이미지
#         분류 : daisy, dandelion, roses, sunflowers, tulips 5가지 분류
import pathlib

from dl.layer7 import optimizers

dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
data_dir = tf.keras.utils.get_file(
    'flower_photos/flower_photos', # 압축 해제 폴더
    origin=dataset_url, # 압축 파일 경로
    untar=True # 압축 해제 여부
)
data_dir = pathlib.Path(data_dir) # 경로 설정
image_count = len(list(data_dir.glob('*/*.jpg'))) # 모든 jpg포맷 이미지의 수
# print(image_count)

# 장미이미지 확인
# PIL 외부라이브러리 설치 : 안될 경우에는 Pillow 설치
roses = list(data_dir.glob('roses/*'))
image = PIL.Image.open(str(roses[0])) # 첫번째 장미이미지
#image.show()

# 데이터셋 만들기

# 훈련셋
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, # 데이터 디렉토리
    validation_split = 0.2, # 검증셋 20%
    subset = 'training', # 서브셋
    seed = 123, # 랜덤시드
    image_size = (180, 180), # 이미지 크기
    batch_size = 32 # batch 크기
)

# 검증셋
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, # 데이터 디렉토리
    validation_split = 0.2, # 검증셋 20%
    subset = 'validation', # 서브셋
    seed = 123, # 랜덤시드
    image_size = (180, 180), # 이미지 크기
    batch_size = 32 # batch 크기
)

# 클래스명
# license.txt 파일이 오인식 되므로 flower_photos는 분류가 아니므로 제거
train_ds.class_names.remove('flower_photos')
# print(class_names) # 5개 분류

# 시각화
# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i+1)
#         plt.imshow(images[i].numpy().astype('uint8'))
#         plt.title(train_ds.class_names[int(labels[i])])
#         plt.axis('off')
# plt.show()

# 데이터 정규화
normalization_layer = layers.Rescaling(1./255) # 0~1 범위로 정규화
# 훈련데이터 각각에 정규화 실시
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# 모델 생성
model = Sequential([
    # 정규화레이어 : 이미지에 있는 색상값(0~255)을 255로 나누면 0~1로 정규화
    # input_shape : 180픽셀(넓이) * 180픽셀(높이) * 3원색(RGB)
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    # 컨볼루션레이어
    # 16 : 16개 윈도우
    # 3 : 3*3크기로 자름
    # padding='same' : 사진크기 유지
    # activation='relu' : 활성화 함수 (출력을 결정하는 함수)
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    # 풀링레이어 : 중요한 정보만 추출
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    # 평탄화레이어 : 여러겹의 이미지를 한 줄로 평탄화해서 표로 구성
    layers.Flatten(),
    # 128개의 뉴런을 사용해서 특성을 출력
    layers.Dense(128, activation='relu'),
    # 5개 중 무엇인지 출력
    layers.Dense(6)
])

# 모델 요약 정보
# model.summary()

# 모델 컴파일
model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
)

# 모델 훈련
history = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 10
)

# 훈련 평가
acc = history.history['accuracy'] # 정확도
val_acc = history.history['val_accuracy'] # 검증 정확도
loss = history.history['loss'] # 손실
val_loss = history.history['val_loss'] # 검증 손실

# 시각화
plt.rcParams['font.family'] = 'Malgun Gothic' # 맑은 고딕 폰트, 그래프내 한글깨짐 방지
plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.plot(range(10), acc, label='훈련 정확도')
plt.plot(range(10), val_acc, label='검증 정확도')
plt.legend(loc='lower right')
plt.title('훈련데이터셋과 검증데이터셋 정확도')

plt.subplot(1, 2, 2)
plt.plot(range(10), loss, label='훈련 손실')
plt.plot(range(10), val_loss, label='검증 손실')
plt.legend(loc='upper right')
plt.title('훈련데이터셋과 검증데이터셋 손실')

plt.show()














































