# layer5.py (완료)
# 손실함수 Categorical Cross Entropy : 다중 클래스 분류용

# 층(layer)을 간단히 순서대로 추가해 모델을 쉽게 구성하게 해주는 클래스
from tensorflow.keras.models import Sequential
# 입력과 출력 뉴런을 연결하는 완전 연결층
from tensorflow.keras.layers import Dense
# 정수형 클래스를 원핫 인코딩으로 변환
from tensorflow.keras.utils import to_categorical
# 배열 연산에 사용되는 핵심 라이브러리
import numpy as np

# 입력 : [공부시간, 수면시간]
X = np.array([[1, 6], [2, 5], [3, 4], [4, 3], [5, 2]])

# 출력(등급) : 0 불합격, 1 보통, 2 합격
y = np.array([0, 0, 1, 2, 2])
y = to_categorical( # 원-핫 인코딩
    y,
    num_classes=3 # 등급의 수 (분류의 수)
)
'''
0 → [1,0,0]
1 → [0,1,0]
2 → [0,0,1]
'''

# 모델 구성
model = Sequential() # 층이 순서대로 추가됨
# 은닉층 : 입력 2개, 뉴런 10개, 활성화 함수 ReLU (비선형성 도입)
model.add(Dense(10, input_dim=2, activation='relu'))
# 출력층 : 출력 클래스 개수만큼 뉴런 생성, softmax(활성화함수) : 출력된 값을 확률로 변환하여 합이 1이 되도록 함. 가장 큰 확률을 가진 클래스가 최종 예측 클래스가 됨
model.add(Dense(3, activation='softmax')) # 분류 수 만큼 출력

# 모델 컴파일 (학습 설정)
model.compile(
    optimizer = 'adam', #최적화 함수 : 손실함수 값을 최소화하기 위해 가중치를 자동으로 조정해주는 알고리즘. 실무에서 가장 많이 사용됨
    loss = 'categorical_crossentropy', #손실함수 : 다중 클래스 분류 문제에서 실제 레이블과 예측 확률 값 간의차이를 측정하는 함수
    metrics=['accuracy']
)

# 모델 학습 : 200회 반복 수행
model.fit(X, y, epochs=200, verbose=0)

# 예측 / 출력
pred = model.predict(np.array([[3, 3]]))
print(f'예측 결과 : {pred}')
print(f'예측 등급 : {np.argmax(pred)}') # 가장 확률 높은 등급 출력
