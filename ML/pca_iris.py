# pca_iris.py
# iris데이터셋을 활용한 PCA알고리즘

# 라이브러리
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 데이터셋
iris = load_iris()

# 독립변수 / 종속변수
X = iris.data # 4차원 데이터
y = iris.target # 품종

# PCA 생성
pca = PCA(n_components=2) # 2차원으로 축소하는 PCA
X_pca = pca.fit_transform(X) # 4차원 > 2차원

# 시각화
plt.rc('font', family='Malgun Gothic')
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA로 축소한 iris데이터셋')
plt.xlabel('첫번째 주성분')
plt.ylabel('두번째 주성분')
plt.colorbar(label='꽃 종류')
plt.show()


























