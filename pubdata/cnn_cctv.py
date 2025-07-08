# cnn_cctv.py
# 스탠포드 자동차 데이터셋
# https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# 데이터 경로 설정
base_dir = './StanfordCars'  # 데이터 압축 해제 위치
train_dir = f'{base_dir}/train'
test_dir = f'{base_dir}/test'

# 클래스 수가 너무 많다면, 예: 처음 10개 클래스만 사용하려면 다음 코드 활성화
import shutil, os
classes = sorted(os.listdir(train_dir))[:10]
for split in ['train', 'test']:
    for cls in os.listdir(f'{base_dir}/{split}'):
        if cls not in classes:
            shutil.rmtree(f'{base_dir}/{split}/{cls}')

# =======================
# 1. 데이터 로딩 및 전처리
# =======================
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    train_dir, # 학습용 이미지가 있는 폴더
    target_size=(128, 128), # 이미지 크기를 128*128
    batch_size=32, # 한번에 32장씩 학습
    class_mode='categorical', # 다중 클래스 분류이므로 one-hot인코딩 사용
    subset='training' # 80%만 학습용으로 사용
)

val_data = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation' # 20%만 검증용으로 사용
)

# =======================
# 2. CNN 모델 구성
# =======================
model = models.Sequential([
    # 첫번째 합성곱 층
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2), # 2x2 풀링
    # 두번째 합성곱 층
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2), # 2x2 풀링
    layers.Flatten(), # 2차원 > 1차원 벡터
    layers.Dense(128, activation='relu'), # 은닉층
    # 클래스 수만큼 출력하는 출력층
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', # 최적화 함수
              loss='categorical_crossentropy', # 손실 함수
              metrics=['accuracy']) # 지표 : 정확도

# =======================
# 3. 모델 학습
# =======================
history = model.fit(
    train_data, # 학습용 이미지
    epochs=10, # 전체 학습 반복 회수
    validation_data=val_data # 검증 데이터 정확도 확인
)

# =======================
# 4. 학습 결과 시각화
# =======================
plt.rc("font", family="Malgun Gothic")
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('정확도 변화')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# =======================
# 5. 테스트 데이터 예측 시각화
# =======================
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=1,
    class_mode='categorical',
    shuffle=True
)

class_names = list(train_data.class_indices.keys()) # 클래스 이름 추출

plt.figure(figsize=(10,10))
for i in range(9): # 예측 결과 시각화 9장
    img, label = next(test_data) # 이미지 1장 가져오기
    pred = model.predict(img) # 예측
    pred_label = class_names[np.argmax(pred[0])] # 예측 클래스
    true_label = class_names[np.argmax(label[0])] # 실제 클래스
    plt.subplot(3,3,i+1)
    plt.imshow(img[0])
    plt.title(f"예측: {pred_label}\n정답: {true_label}",
              color='green' if pred_label==true_label else 'red')
    plt.axis('off')
plt.tight_layout()
plt.show()
































