# ex-knn.py
# KNN 실습 풀이

'''
실습 주제 : 의류 이미지 속성 분류 (패션 MNIST / Fashion MNIST)
실습 설명 : Kaggle의 Fashion MNIST데이터셋 흑백 옷 이미지(28x28픽셀)를
           바탕으로 옷의 종류(티셔츠, 바지, 신발 등)를 분류
데이터셋 출처 : https://www.kaggle.com/datasets/zalando-research/fashionmnist
                   (fashion-mnist_train.csv, fashion-mnist_test.csv)
'''

# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 1. 데이터셋 로딩
df = pd.read_csv('assets/fashion-mnist_train.csv')
#df.info()

# 2. X(독립변수), y(종속변수) 분리
X = df.drop('label', axis=1)  # 픽셀 데이터 (784개 컬럼)
y = df['label']               # 정답 클래스 (0~9 숫자)

# 클래스 번호를 실제 의미로 매핑하기 위한 딕셔너리
label_names = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# 3. 전체 데이터 중에서 5000개만 사용
X = X[:5000]
y = y[:5000]

# 4. 훈련/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. KNN 모델 생성 / 학습
model = KNeighborsClassifier(n_neighbors=3)  # 이웃 수 K = 3
model.fit(X_train, y_train)  # 훈련 데이터로 모델 학습

# 6. 테스트 데이터 예측
y_pred = model.predict(X_test)

# 7. 정확도 및 정밀도, 재현율, F1-score 출력
print("정확도:", accuracy_score(y_test, y_pred))
print("\n분류 리포트:\n", \
    classification_report(
          y_test,
          y_pred,
          target_names=[label_names[i] for i in sorted(label_names)]
    )
)

# 8. 예측 시각화 (9개만)
plt.figure(figsize=(10, 5))
plt.rc('font', family='Malgun Gothic')
for i in range(9):
    plt.subplot(3, 3, i+1)
    image = X_test.iloc[i].values.reshape(28, 28)  # 1차원 → 28x28 이미지
    plt.imshow(image, cmap='gray')
    plt.title(f"실제: {label_names[y_test.iloc[i]]}\n예측: {label_names[y_pred[i]]}")
    plt.axis('off')

plt.suptitle('KNN 분류 결과 예시 (예측 클래스)')
plt.tight_layout()
plt.show()










































