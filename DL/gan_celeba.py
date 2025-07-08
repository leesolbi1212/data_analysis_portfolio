# ✅ 1. 필수 라이브러리 불러오기
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import urllib.request
from PIL import Image
import glob

# ✅ 2. 정상 작동하는 URL에서 CelebA Tiny (28x28, 2000장) 다운로드
celeba_url = "https://github.com/inktokyo/public-datasets/releases/download/celeba-tiny/img_align_celeba_28x28.zip"
zip_path = "./celeba/img_align_celeba_28x28.zip"

if not os.path.exists(zip_path):
    print("📥 Downloading CelebA Tiny dataset...")
    urllib.request.urlretrieve(celeba_url, zip_path)

# ✅ 3. 압축 해제
if not os.path.exists("./celeba/img_align_celeba"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("./celeba/")

# ✅ 4. 이미지 전처리 함수
def load_images(img_folder, img_size=(28, 28), max_images=2000):
    image_paths = glob.glob(os.path.join(img_folder, "*.jpg"))[:max_images]
    data = []
    for path in image_paths:
        img = Image.open(path).resize(img_size)
        img = np.asarray(img) / 127.5 - 1.0
        if img.shape == (28, 28, 3):
            data.append(img)
    return np.array(data, dtype=np.float32)

# ✅ 5. 데이터셋 로딩 및 구성
images = load_images("./celeba/img_align_celeba", max_images=2000)
dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(2000).batch(256)

# ✅ 6. Generator 모델
def make_generator():
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(100,)),
        layers.Dense(28 * 28 * 3, activation='tanh'),
        layers.Reshape((28, 28, 3))
    ])
    return model

# ✅ 7. Discriminator 모델
def make_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 3)),
        layers.Dense(128),
        layers.LeakyReLU(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# ✅ 8. 인스턴스 생성 및 옵티마이저
generator = make_generator()
discriminator = make_discriminator()

cross_entropy = tf.keras.losses.BinaryCrossentropy()
gen_opt = tf.keras.optimizers.Adam(1e-4)
disc_opt = tf.keras.optimizers.Adam(1e-4)

# ✅ 9. 손실 함수
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# ✅ 10. 학습 Step 정의
@tf.function
def train_step(images):
    noise = tf.random.normal([256, 100])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(fake_images, training=True)

        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)

    gen_opt.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    disc_opt.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# ✅ 11. 시각화 함수
def generate_and_show_images(generator, epoch, seed):
    predictions = generator(seed, training=False)
    predictions = (predictions + 1) / 2.0

    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis("off")
    plt.suptitle(f"Epoch {epoch}")
    plt.tight_layout()
    plt.show()

# ✅ 12. 학습 루프
def train(dataset, epochs):
    seed = tf.random.normal([16, 100])
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        print(f"Epoch {epoch+1} 완료 ✅")
        generate_and_show_images(generator, epoch+1, seed)

# ✅ 13. 학습 시작
train(dataset, epochs=10)
