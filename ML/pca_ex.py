# pca_ex.py
# 실습 : iris데이터셋을 활용한 PCA알고리즘 

# 3차원으로 축소해서 시각화 해보기

# 라이브러리
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 데이터셋
iris = load_iris()

# 독립변수 / 종속변수
X = iris.data # 4차원 데이터
y = iris.target # 품종
target_names = iris.target_names

# PCA 생성
pca = PCA(n_components=3) # 3차원으로 축소하는 PCA
X_pca = pca.fit_transform(X) # 4차원 > 3차원

# 시각화
plt.rc('font', family='Malgun Gothic')
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 클래스 별로 색상 다르게 시각화
colors = ['red', 'green', 'blue']
for i, target_name in enumerate(target_names):
    print(i)
    ax.scatter(
        X_pca[y == i, 0], # setosa
        X_pca[y == i, 1], # versicolor
        X_pca[y == i, 2], # virginica
        label=target_name,
        color=colors[i],
        edgecolors='k',
        alpha=0.8
    )

ax.set_title("Iris 데이터의 PCA 3차원 시각화")
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.legend()
plt.show()


























