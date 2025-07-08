# tf_mnist.py
# 텐서플로우를 활용한 모델 구성 및 컴파일 과정

# 라이브러리
import tensorflow as tf
import matplotlib.pyplot as plt

# 데이터셋
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 데이터 정규화 (0 ~ 1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Sequential 모델 구성
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # 입력층 : 28*28 => 784
    tf.keras.layers.Dense(128, activation='relu'), # 은닉층 : 128개 뉴런
    tf.keras.layers.Dense(10, activation='softmax') # 출력층 : 10개 클래스
])

# 모델 컴파일
model.compile(
    optimizer = 'adam', # 최적화 함수
    loss = 'sparse_categorical_crossentropy', # 손실 함수
    metrics = ['accuracy'] # 정확도
)

# 모델 학습
model.fit(X_train, y_train, epochs=5) # epochs : 학습 회수

# 모델 평가
test_loss, test_acc = model.evaluate(X_test, y_test) # 테스트손실, 테스트정확도
print(f'테스트 정확도 : {test_acc:.4f}')

# 예측
predictions = model.predict(X_test)

# 시각화 함수
def plot_image(i, predictions_array, true_label, img):
    plt.grid(False) # 그리드 제거
    plt.xticks([]) # X축 눈금값 제거
    plt.yticks([]) # Y축 눈금값 제거
    plt.imshow(img, cmap=plt.cm.binary) # 이미지를 흑백으로 출력
    predicted_label = tf.argmax(predictions_array) # 확률이 높은 클래스를 반환
    true = true_label[i] # 현재 정답 레이블
    # 예측이 정답이면 파란색, 그렇지 않으면 빨간색
    color = 'blue' if predicted_label==true else 'red'
    # 예측된 분류와 정답을 X축 레이블로
    plt.xlabel(f'예측:{predicted_label.numpy()}, 정답:{true}', color=color)

# 이미지 10개 출력
plt.rc('font', family='Malgun Gothic')
plt.figure(figsize=(16, 12))
for i in range(160):
    plt.subplot(16, 10, i+1)
    plot_image(i, predictions[i], y_test, X_test[i])
plt.tight_layout()
plt.show()
























