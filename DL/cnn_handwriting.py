# cnn_handwriting.py
# cnn을 활용한 손글씨 이미지 인식

# 라이브러리
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from PIL import Image
import cv2 # opencv-python 외부라이브러리 설치
import os

# mnist 데이터로 CNN 모델 학습
def build_and_train_model():
    # 훈련/테스트 분리
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # 정규화
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    # Sequential 모델
    model = models.Sequential([
        layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ])
    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # 모델 훈련
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    # 모델 저장
    model.save("mnist_digit_model.h5")
    return model

# 모델 불러오기
def load_model(force_train=False):
    if force_train or not os.path.exists("mnist_digit_model.h5"):
        return build_and_train_model()
    return tf.keras.models.load_model("mnist_digit_model.h5")
# 이미지 전처리
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # 흑백 변환
    img = img.resize((28, 28))  # MNIST input size
    img_np = np.array(img)

    # 흰 배경일 경우 반전
    if np.mean(img_np) > 127:
        img_np = 255 - img_np

    img_np = img_np / 255.0  # 정규화
    img_np = img_np.reshape(1, 28, 28, 1).astype("float32")
    return img_np, img

# 예측 / 시각화
def predict_and_visualize(image_path, model):
    preprocessed_img, raw_img = preprocess_image(image_path)
    logits = model.predict(preprocessed_img)
    pred = np.argmax(logits)

    plt.rc('font', family='Malgun Gothic')
    plt.imshow(raw_img, cmap='gray')
    plt.title(f"예측된 숫자: {pred}")
    plt.axis('off')
    plt.show()

# 실행
if __name__ == '__main__': # 메인모듈이라면
    model = load_model(force_train=True) # 모델 로딩
    image_path = 'assets/number.png'
    predict_and_visualize(image_path, model)

























