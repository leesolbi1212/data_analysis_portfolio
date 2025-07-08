# cnn_cifar10.py
# CNN을 활용한 이미지 분류

# 라이브러리 임포트
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# mnist데이터셋
# 10개 클래스, 6만개 컬러이미지(5만개 훈련용, 1만개 테스트용)
(train_images, train_labels), (test_images, test_labels) = \
    datasets.cifar10.load_data()

# 데이터 정규화 : 모든 픽셀값들이 0에서 1사이에 위치하도록
train_images, test_images = train_images/255.0, test_images/255.0

# 클래스명 10개
class_names = ['airplane', 'automobile', 'bird', 'cat', \
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 데이터 시각화
# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1) # 5행 5열, 순서번호
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i])
#     plt.xlabel(class_names[test_labels[i][0]])
# plt.show()

# 모델 생성, 합성곱레이어 생성

# 시퀀셜 모델 : 각 층들을 추가된 순서대로 처리하는 모델
model = models.Sequential()
# 합성곱레이어 추가
model.add(
    layers.Conv2D(
        32, # 필터 개수 : 32개의 필터를 적용
        (3,3), # 커널(이미지를 분석할 범위) 크기
        activation = 'relu', # 활성화 함수
        input_shape = (32, 32, 3) # 입력 형태 : (높이, 너비, 채널 수)
    )
)
# MaxPooling2D 레이어 추가
model.add(
    layers.MaxPooling2D(
        (2, 2) # 풀링 창 크기 : 2x2영역에서 최대값 추출
    )
)
model.add(
    layers.Conv2D(
        64,
        (3, 3),
        activation = 'relu'
    )
)
model.add(
    layers.MaxPooling2D(
        (2, 2)
    )
)
model.add(
    layers.Conv2D(
        64,
        (3, 3),
        activation = 'relu'
    )
)

# 전체 흐름
'''
Conv2D     : 출력채널수(32), 커널(3,3), 풀링X, 출력크기 32x32 => 30x30
MaxPooling :               커널(2,2), 풀링O, 출력크기 30x30 => 15x15
Conv2D     : 출력채널수(64), 커널(3,3), 풀링X, 출력크기 15x15 => 13x13
MaxPooling :               커널(2,2), 풀링O, 출력크기 13x13 => 6x6
Conv2D     : 출력채널수(64), 커널(3,3), 풀링X, 출력크기 6x6 => 4x4
'''

# 모델 요약정보
# model.summary()

# 모델에 Flatten, Dense 레이어 추가
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()

# 모델 컴파일
model.compile(
    optimizer = 'adam', # 활성화 함수
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # 손실함수
    metrics = ['accuracy'] # 정확도 지표
)

# 모델 훈련
history = model.fit(
    train_images,
    train_labels,
    epochs = 20, # 훈련 회수
    validation_data = (test_images, test_labels) # 검증 데이터
)

# 모델 평가 / 시각화
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1]) # y축 값의 범위
plt.legend(loc = 'lower right') # 범례 위치
# verbose : 출력로그 수준, 0:출력안함, 1:진행바출력, 2:epoch별로한줄출력
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
plt.show()

# 정확도, 손실 출력
print(f'정확도:{test_acc}, 손실:{test_loss}')









































