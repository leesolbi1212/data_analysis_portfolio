# cnn_handwriting.py
# cnn을 활용한 손글씨 이미지 인식
# MNIST 데이터로 학습하고, 이미지 파일을 입력받아 숫자를 예측

# 라이브러리
import tensorflow as tf  # 딥러닝 프레임워크 (모델 생성, 학습, 예측)
import numpy as np # 수치 연산 및 배열 조작
import matplotlib.pyplot as plt # 이미지 시각화
from tensorflow.keras import layers, models # 케라스의 층(Layer), 모델(Model) 관련 모듈
from PIL import Image # 이미지 파일을 열고 처리하기 위한 모듈 (Pillow)
import cv2 # opencv-python 외부라이브러리 설치  # OpenCV: 이미지 처리용 외부 라이브러리 (향후 고급 기능에 사용 가능)
import os  # 파일 존재 여부 확인 등 시스템 관련 기능

# [1] MNIST 데이터셋으로 CNN 모델을 학습하고 저장하는 함수
def build_and_train_model():
    # MNIST 손글씨 데이터셋을 불러옴 (28x28 크기 흑백 이미지, 0~9 라벨)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 훈련/테스트 이미지 데이터를 4차원 텐서로 reshape + 정규화(0~1 범위로 스케일링)
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # CNN 모델 정의 (Sequential 방식으로 층을 순차적으로 쌓음)
    model = models.Sequential([
        layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1)),  # 3x3 필터 16개, 입력 크기 지정
        layers.MaxPooling2D(), # 2x2 최대 풀링으로 이미지 크기 절반 축소
        layers.Conv2D(32, 3, activation='relu'), # 3x3 필터 32개
        layers.MaxPooling2D(), # 다시 2x2 풀링
        layers.Conv2D(64, 3, activation='relu'), # 3x3 필터 64개
        layers.MaxPooling2D(), # 3번째 풀링
        layers.Flatten(), # 3차원 feature map을 1차원 벡터로 평탄화
        layers.Dense(64, activation='relu'), # 은닉층 (뉴런 64개, ReLU 활성화 함수)
        layers.Dense(10)  # 출력층 (0~9 숫자 분류, softmax 없이 로짓 그대로 출력)
    ])
    # 모델 컴파일 (옵티마이저: Adam, 손실함수: 다중분류용, 평가지표: 정확도)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    # 모델 학습 (에폭 10회 반복, 검증 데이터로 테스트셋 사용)
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    # 학습된 모델 저장 (파일로 저장하여 다음 실행 시 재사용 가능)
    model.save("mnist_digit_model.h5")
    # 학습 완료된 모델 반환
    return model

# [2] 모델을 불러오는 함수 (없으면 새로 학습하고 저장함)
def load_model(force_train=False):
    # force_train=True 또는 모델 파일이 없으면 새로 학습
    if force_train or not os.path.exists("mnist_digit_model.h5"):
        return build_and_train_model()
    # 저장된 모델이 있으면 불러오기
    return tf.keras.models.load_model("mnist_digit_model.h5")

# [3] 이미지 전처리 함수
def preprocess_image(image_path):
    # 이미지 열기 + 흑백 모드로 변환 (L 모드는 흑백)
    img = Image.open(image_path).convert('L')
    # MNIST 사이즈인 28x28 픽셀로 리사이즈
    img = img.resize((28, 28))
    # 이미지 데이터를 numpy 배열로 변환
    img_np = np.array(img)

    # 흰 배경 이미지일 경우 숫자 색상 반전 (MNIST는 검은 배경 + 흰 숫자)
    if np.mean(img_np) > 127: # 평균 픽셀값이 밝으면 → 반전
        img_np = 255 - img_np

    # 0~1 사이로 정규화 + CNN 입력 형식 (4차원)으로 변형
    img_np = img_np / 255.0  # 정규화
    img_np = img_np.reshape(1, 28, 28, 1).astype("float32")
    # 전처리된 이미지와 원본 이미지(PIL) 둘 다 반환
    return img_np, img

# [4] 예측 및 시각화 함수
def predict_and_visualize(image_path, model):
    # 이미지 전처리
    preprocessed_img, raw_img = preprocess_image(image_path)
    # 모델로 예측 수행 (출력은 10개의 로짓 값)
    logits = model.predict(preprocessed_img)
    # 로짓 중 가장 큰 값을 갖는 인덱스 → 예측된 숫자 클래스
    pred = np.argmax(logits)

    plt.rc('font', family='Malgun Gothic')
    plt.imshow(raw_img, cmap='gray')
    plt.title(f"예측된 숫자: {pred}")
    plt.axis('off')
    plt.show()

# [5] 실행 부분 (스크립트를 직접 실행했을 때만 아래 코드 실행)
if __name__ == '__main__': # 메인모듈이라면
    # 모델을 로딩하거나 학습함 (force_train=True: 무조건 새로 학습)
    model = load_model(force_train=True)
    # 예측할 이미지 경로 지정 (미리 준비한 손글씨 이미지 파일)
    image_path = 'assets/number1.png'
    # 예측 및 결과 시각화
    predict_and_visualize(image_path, model)

