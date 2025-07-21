# cnn_cifar10.py
# CNN을 활용한 이미지 분류

# 라이브러리 임포트
import tensorflow as tf # 구글이 만든 딥러닝 라이브러리. tf라는 별칭으로 불러와서 사용
from tensorflow.keras import datasets, layers, models
# datasets : 미리 준비된 데이터를 쉽게 불러오는 모듈
# layers : 신경망 층(layer)을 정의하는 모듈
# models : 층을 모아 모델을 만드는 모듈
import matplotlib.pyplot as plt

# mnist데이터셋
# 10개 클래스, 6만개 컬러이미지 (5만개 훈련용, 1만개 테스트용)
(train_images, train_labels), (test_images, test_labels) = \
    datasets.cifar10.load_data()

# 데이터 정규화 : 모든 픽셀값들이 0에서 1사이에 위치하도록
# 이미지 픽셀값은 0~255 범위의 정수 -> 0~1 사이로 바꾸기 (정규화)
# 255.0으로 나누면 실수(float)로 변환되고, 신경망 학습 시 수치 안정성과 속도가 좋아짐.
train_images, test_images = train_images/255.0, test_images/255.0

# 클래스명 10개
# 숫자 레이블을 사람이 읽을 수 있는 문자열로 매핑하기 위함
class_names = ['airplane', 'automobile', 'bird', 'cat', \
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 데이터 시각화
# plt.figure(figsize=(10, 10)) # 전체 그림판 크기를 10×10인치로 설정
# for i in range(25): # 처음 25개 이미지를 한 번에 살펴보기 위해 반복문 시작
#     plt.subplot(5, 5, i+1) # 5행 5열 중 i+1번째 칸 순서 번호
#     plt.xticks([])  # x축 눈금 없애기
#     plt.yticks([])  # y축 눈금 없애기
#     plt.grid(False)   # 격자선 없애기
#     plt.imshow(train_images[i]) # i번째 훈련 이미지 표시
#     plt.xlabel(class_names[train_labels[i][0]])
# plt.show()

# 모델 생성, 합성곱 레이어 생성

# 시퀀셜 모델 : 각 층들을 추가된 순서대로 처리하는 모델
model = models.Sequential()
# 합성곱레이어 추가
model.add(
    layers.Conv2D(
        32, # 필터 개수 : 32개의 필터를 적용
        (3,3), # 커널(이미지를 분석할 범위) 크기, 필터 윈도우
        activation = 'relu', # 활성화 함수
        input_shape = (32, 32, 3) # 입력 형태 : (높이, 너비, 채널 수)
    )
)
# MaxPooling2D 레이어 추가
model.add(
    layers.MaxPooling2D(
        (2, 2) # 풀링 창 크기 : 영역마다 최대값만 추출해 이미지 크기를 절반(예: 30→15)로 줄임
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
# 각 층의 출력 크기, 파라미터 수 등을 테이블 형태로 출력
# 전체 모델 크기와 복잡도를 한 눈에 파악할 수 있음
# model.summary()

# 모델에 Flatten, Dense 레이어 추가
model.add(layers.Flatten()) # 다차원 텐서를 1차원 벡터로 펼쳐서 Dense 층에 전달
model.add(layers.Dense(64, activation='relu')) # Dense 완전 연결층, 뉴런 64개 ReLU 활성화 함수
model.add(layers.Dense(10)) # 최종 출력 층 : 클래스 수 (10개)만큼 뉴런, softmax를 나중에 적용할 수 있도록 여기서는 로짓(정수 점수)만 출력

# model.summary()

# 모델 컴파일
model.compile(
    optimizer = 'adam', # 활성화 함수 (학습률 자동 조정, 대중적으로 좋은 성능)
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
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1]) # y축 값의 범위
# plt.legend(loc = 'lower right') # 범례 위치
# # verbose : 출력로그 수준, 0:출력안함, 1:진행바출력, 2:epoch별로한줄출력
# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# # plt.show()

# 정확도, 손실 출력
# print(f'정확도:{test_acc}, 손실:{test_loss}')

import matplotlib.pyplot as plt
import  numpy as np

# 랜덤으로 25개 샘플 선택
idxs = np.random.choice(len(test_images), size=25, replace=False)

plt.figure(figsize=(10,10))
for i, idx in enumerate(idxs):
    plt.subplot(5,5,i+1)
    plt.xticks([]); plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[idx])
    true = class_names[true_labels[idx]]
    pred = class_names[predictions[idx]]
    color = 'green' if true == pred else 'red'
    plt.xlabel(f'T:{true}\nP:{pred}', color=color)
plt.tight_layout()
plt.show()
