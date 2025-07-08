# classification-knn.py
# KNN 분류

# 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets # 데이터셋
from sklearn.neighbors import KNeighborsClassifier # KNN 분류기
from sklearn.model_selection import train_test_split # 훈련/테스트셋 분리
from sklearn.decomposition import PCA # 차원축소

# iris 데이터셋
iris = datasets.load_iris()

# 데이터와 레이블 분리
X = iris.data # 독립변수 (150개 데이터, 4개 특성)
y = iris.target # 종속변수 (클래스 번호 0 또는 1 또는 2)

# 훈련/테스트 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

# KNN 분류기 (이웃의 개수 K를 3으로 설정)
knn = KNeighborsClassifier(n_neighbors=3)

# 모델 학습
knn.fit(X_train, y_train)

# 예측 수행
y_pred = knn.predict(X_test)

# 예측 정확도
accuracy = np.mean(y_pred == y_test)
print(f'정확도 : {accuracy:.2f}')

# 4차원 데이터를 2차원으로 차원 축소
pca = PCA(n_components=2) # 2차원 차원축소기
X_reduced = pca.fit_transform(X) # 전체 데이터를 2차원으로 축소

# 축소된 데이터를 훈련/테스트 분리
X_train_reduced, X_test_reduced, _, _ = train_test_split(
    X_reduced,
    y,
    test_size=0.2,
    random_state=42
)

# 분류 결과 시각화
plt.figure(figsize=(10, 6))
plt.rc('font', family='Malgun Gothic')
plt.scatter(
    X_train_reduced[:, 0],
    X_train_reduced[:, 1],
    c=y_train,
    cmap='viridis',
    marker='o',
    label='훈련데이터',
    edgecolor='k'
)
plt.scatter(
    X_test_reduced[:, 0],
    X_test_reduced[:, 1],
    c=y_pred,
    cmap='coolwarm',
    marker='^',
    label='예측결과',
    edgecolor='k'
)
plt.title('KNN 분류 결과 시각화')
plt.xlabel('PCA 컴포넌트 1')
plt.ylabel('PCA 컴포넌트 2')
plt.legend()
plt.grid(True)
plt.show()






































