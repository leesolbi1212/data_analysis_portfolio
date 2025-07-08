# gan_cifar10.py
# CIFAR-10 이미지를 생성 및 분류하는 GAN 모델

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# 데이터셋 로드 및 전처리
(X_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
X_train = (X_train.astype('float32') - 127.5) / 127.5
y_train = y_train.flatten()

BUFFER_SIZE = 50000
BATCH_SIZE = 128
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

EPOCHS = 50
NOISE_DIM = 100
NUM_CLASSES = 10

# Generator 모델 정의
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(NOISE_DIM,)))
    model.add(layers.Dense(8 * 8 * 256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8, 8, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

# Discriminator 모델 정의
class DualDiscriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.Sequential([
            tf.keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
            layers.LeakyReLU(),
            layers.Dropout(0.3),
            layers.Flatten(),
        ])
        self.real_fake_output = layers.Dense(1, activation='sigmoid')
        self.class_output = layers.Dense(NUM_CLASSES, activation='softmax')

    def call(self, x, training=False):
        x = self.conv(x, training=training)
        return self.real_fake_output(x), self.class_output(x)

# 모델 생성 및 초기화
generator = make_generator_model()
discriminator = DualDiscriminator()
_ = generator(tf.random.normal([1, NOISE_DIM]))
_ = discriminator(tf.random.normal([1, 32, 32, 3]))

# 손실함수 및 옵티마이저 정의
cross_entropy = tf.keras.losses.BinaryCrossentropy()
class_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# 학습 step
@tf.function
def train_step(images, labels):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    fake_labels = tf.random.uniform([BATCH_SIZE], 0, NUM_CLASSES, dtype=tf.int32)

    with tf.GradientTape(persistent=True) as tape:
        fake_images = generator(noise, training=True)
        real_output, real_class = discriminator(images, training=True)
        fake_output, fake_class = discriminator(fake_images, training=True)

        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        class_loss_real = class_loss_fn(labels, real_class)
        class_loss_fake = class_loss_fn(fake_labels, fake_class)
        d_loss = real_loss + fake_loss + class_loss_real + class_loss_fake

        gen_rf_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        gen_class_loss = class_loss_fn(fake_labels, fake_class)
        g_loss = gen_rf_loss + gen_class_loss

    gradients_of_generator = tape.gradient(g_loss, generator.trainable_variables)
    gradients_of_discriminator = tape.gradient(d_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# 이미지 생성 및 시각화
def generate_and_show_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = (predictions + 1) / 2.0

    fig = plt.figure(figsize=(10, 2))
    for i in range(predictions.shape[0]):
        plt.subplot(2, 5, i+1)
        plt.imshow(predictions[i])
        plt.axis('off')
    plt.suptitle(f"Generated Samples (Epoch {epoch})")
    plt.tight_layout()
    plt.show()

# 전체 학습 루프
def train(dataset, epochs):
    seed = tf.random.normal([10, NOISE_DIM])
    for epoch in range(epochs):
        for images, labels in dataset:
            train_step(images, labels)
        print(f"Epoch {epoch+1} 완료")
        generate_and_show_images(generator, epoch+1, seed)

# 학습 시작
train(train_dataset, EPOCHS)


# # gan_cifar10.py
# # GAN을 사용하여 CIFAR-10 사물 이미지를 생성하고
# # 진위를 판별하고 사물의 분류를 수행
#
# # 라이브러리
# import tensorflow as tf
# from tensorflow.keras import layers
# import numpy as np
# import matplotlib.pyplot as plt
# #from tensorflow.python.ops.distributions.kullback_leibler import cross_entropy
#
# # 데이터셋
# (X_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
#
# # 이미지 정규화 : 픽셀값을 [-1, 1] 범위로 조정 (tanh 출력에 맞추기 위함)
# X_train = (X_train.astype('float32') - 127.5) / 127.5
# #print(X_train)
#
# # 라벨을 1차원 배열로 평탄화
# y_train = y_train.flatten()
# #print(y_train)
#
# # 학습용 배치
# BUFFER_SIZE = 50000 # 데이터 셔플 크기
# BATCH_SIZE = 128 # 한번에 학습할 배치 크기
# train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
#     .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# #print(train_dataset) # shape : (32, 32, 3)
#
# # 하이퍼파라미터
# EPOCHS = 50
# NOISE_DIM = 100
# NUM_CLASSES = 10
# CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', \
#                'dog', 'frog', 'horse', 'ship', 'truck']
#
#
# def make_generator_model():
#     model = tf.keras.Sequential()
#
#     # 명시적으로 입력층 정의
#     model.add(tf.keras.Input(shape=(NOISE_DIM,)))
#     model.add(layers.Dense(8 * 8 * 256, use_bias=False))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())
#     model.add(layers.Reshape((8, 8, 256)))
#
#     model.add(layers.Conv2DTranspose(
#         128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())
#
#     model.add(layers.Conv2DTranspose(
#         64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
#     model.add(layers.BatchNormalization())
#     model.add(layers.LeakyReLU())
#
#     model.add(layers.Conv2DTranspose(
#         3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
#
#     return model
#
# # # Generator 모델
# # # 노이즈 벡터를 받아서 32 X 32 X 3 크기의 컬러 이미지 생성
# # # 입력 : 100차원 노이즈 벡터
# # # 출력 : 32 X 32 X 3 크기의 이미지 (픽셀값은 [-1, 1] 범위)
# # def make_generator_model():
# #     model = tf.keras.Sequential() # 시퀀셜 모델
# #
# #     # 100차원 노이즈 : 8 X 8 X 256
# #     # 입력 : 100차원 노이즈
# #     # 출력 : (batch_size, 16384)
# #     # 노이즈를 선형 변환해서 특징 맵으로 확장
# #     # 파라미터 : use_bias=False : 배치정규화를 수행하므로 bias 생략
# #     #           input_shape : 100차원 노이즈 벡터
# #     model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(NOISE_DIM,)))
# #     # 학습 초기 불안정성 감소해서 학습속도 향상, 평균 0이고 분산 1로 정규화
# #     model.add(layers.BatchNormalization()) # 학습안정화를 위한 정규화
# #     # 비선형성 추가, 음수 영역도 완전히 죽지 않도록 작은 기울기 유지
# #     model.add(layers.LeakyReLU()) # 비선형성
# #     # 1D 벡터 > 3D 텐서, 8X8픽셀에 256개의 채널
# #     model.add(layers.Reshape((8, 8, 256))) # 텐서를 3D 이미지형으로 변환
# #
# #     # 업샘플링 : 8 X 8 X 256 > 8 X 8 X 128
# #     model.add(layers.Conv2DTranspose(
# #         128, (5,5), strides=(1,1), padding='same', use_bias=False)
# #     )
# #     model.add(layers.BatchNormalization())
# #     model.add(layers.LeakyReLU())
# #
# #     # 업샘플링 : 8 X 8 X 128 > 16 X 16 X 64
# #     model.add(layers.Conv2DTranspose(
# #         64, (5,5), strides=(2,2), padding='same', use_bias=False)
# #     )
# #     model.add(layers.BatchNormalization())
# #     model.add(layers.LeakyReLU())
# #
# #     # 업샘플링 : 16 X 16 X 64 > 32 X 32 X 3
# #     # 3: 출력채널 수
# #     # 활성화함수 tanh 사용 : 픽셀값을 [-1, 1] 범위로 제한
# #     # 출력 : 32 X 32 X 3 컬러 이미지
# #     model.add(layers.Conv2DTranspose(
# #         3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')
# #     )
# #
# #     return model
#
# # Discriminator 모델
# # 입력 : 32 X 32 X 3 입력 이미지
# # 진짜/가짜 판별, 10개 클래스 분류
# class DualDiscriminator(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.conv = tf.keras.Sequential([
#             layers.Conv2D(
#                 64, # 출력 채널 수
#                 (5,5), # 커널 크기
#                 strides=(2,2), # 다운샘플링 : 2배 줄이기 (32X32 > 16X16)
#                 padding='same', # 출력 크기를 유지
#                 input_shape=(32,32,3) # 입력 이미지
#             ),
#             layers.LeakyReLU(), # 비선형성 추가 (음수영역도 반영)
#             layers.Dropout(0.3), # 과적합방지를 위해 30%를 무작위로 제거
#             layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
#             layers.LeakyReLU(),
#             layers.Dropout(0.3),
#             layers.Flatten() # 1D 벡터로 변환
#         ])
#         self.real_fake_output = layers.Dense(
#             1, # 이진 분류 (하나의 확률값 출력)
#             activation='sigmoid' # 0~1 확률 범위
#         )
#         self.class_output = layers.Dense(
#             NUM_CLASSES, # 클래스의 수
#             activation='softmax' # 각 클래스 확률로 출력 (총합이 1)
#         )
#
#     # x : 입력 이미지(32X32X3)
#     # training : 학습 시 True, 추론 시 False
#     # 반환 값 : 진짜/가짜 확률, 10 클래스 확률 벡터
#     def call(self, x, training=False):
#         x = self.conv(x, training=False) # CNN 블록을 통해 특징 추출
#         return self.real_fake_output(x), self.class_output(x)
#
# # 모델 생성
# generator = make_generator_model() # 생성자 모델
# discriminator = DualDiscriminator() # 판별자 모델
#
# # 모델 변수 초기화를 위한 더미 입력
# _ = generator(tf.random.normal([1, NOISE_DIM]))
# _ = discriminator(tf.random.normal([1, 32, 32, 3]))
#
# # 손실 함수 및 최적화함수
# cross_entropy = tf.keras.losses.BinaryCrossentropy() # 진짜/가짜 판별용 이진 손실
# class_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy() # 클래스 분류용 손실
# generator_optimizer = tf.keras.optimizers.Adam(1e-4) # 생성자 옵티마이저
# discriminator_optimizer = tf.keras.optimizers.Adam(1e-4) # 판별자 옵티마이저
#
# # 학습 step 정의
# #@tf.function
# def train_step(images, labels):
#     noise = tf.random.normal([BATCH_SIZE, NOISE_DIM]) # 랜덤 노이즈 생성
#     fake_images = generator(noise, training=True) # 가짜 이미지 생성
#     fake_labels = tf.random.uniform([BATCH_SIZE], 0, NUM_CLASSES, dtype=tf.int32)  # 가짜 클래스 라벨 생성
#
#     with tf.GradientTape(persistent=True) as tape:
#         real_output, real_class = discriminator(images, training=True)   # 진짜 이미지 결과
#         fake_output, fake_class = discriminator(fake_images, training=True) # 가짜 이미지 결과
#
#         # 판별자 손실 계산
#         real_loss = cross_entropy(tf.ones_like(real_output), real_output) # 진짜는 1
#         fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output) # 가짜는 0
#         class_loss_real = class_loss_fn(labels, real_class) # 진짜 이미지 클래스 분류 손실
#         class_loss_fake = class_loss_fn(fake_labels, fake_class) # 가짜 이미지 클래스 분류 손실
#         d_loss = real_loss + fake_loss + class_loss_real + class_loss_fake
#
#         # 생성자 손실 계산
#         gen_rf_loss = cross_entropy(tf.ones_like(fake_output), fake_output) # 가짜를 진짜처럼 보이게
#         gen_class_loss = class_loss_fn(fake_labels, fake_class) # 원하는 클래스처럼 보이게
#         g_loss = gen_rf_loss + gen_class_loss
#
#     # 가중치 업데이트
#     gradients_of_generator = tape.gradient(g_loss, generator.trainable_variables)
#     gradients_of_discriminator = tape.gradient(d_loss, discriminator.trainable_variables)
#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
#
# # 시각화 함수
# def generate_and_show_images(model, epoch, test_input):
#     predictions = model(test_input, training=False)
#     predictions = (predictions + 1) / 2.0  # 이미지 복원 ([-1,1] → [0,1])
#
#     fig = plt.figure(figsize=(10, 2))
#     for i in range(predictions.shape[0]):
#         plt.subplot(2, 5, i+1)
#         plt.imshow(predictions[i])
#         plt.axis('off')
#     plt.suptitle(f"Generated Samples (Epoch {epoch})")
#     plt.tight_layout()
#     plt.show()
#
# # 전체 학습 루프
# def train(dataset, epochs):
#     seed = tf.random.normal([10, NOISE_DIM])  # 고정된 노이즈 벡터
#     for epoch in range(epochs):
#         for images, labels in dataset:
#             train_step(images, labels)
#         print(f"Epoch {epoch+1} 완료")
#         generate_and_show_images(generator, epoch+1, seed)
#
# # 학습 실행
# train(train_dataset, EPOCHS)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
