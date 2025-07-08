# clustering.py
# 클러스터링

# 1. K-Means 클러스터링

# 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')
from sklearn.datasets import make_blobs # 무작위 데이터셋 생성용

# random seed 설정
np.random.seed(42)

# 중심점이 4개인 150개의 점 데이터를 무작위 생성
points, labels = make_blobs(
    n_samples=150, # 샘플데이터 수
    centers=4, # 중심점 수
    n_features=2, # feature 수
    random_state=42 # 랜덤시드값
)
#print(points.shape, '\n', points[:10])
#print(labels.shape, '\n', labels[:10])

# 데이터 프레임
points_df = pd.DataFrame(points, columns=['X', 'Y'])
#print(points_df)

# 스캐터 그리기
# figure = plt.figure(figsize=(10, 6))
# axes = figure.add_subplot(111)
# axes.scatter(points_df['X'], points_df['Y'], label='Random Data')
# axes.grid()
# axes.legend()
# plt.show()

# K-Means 라이브러리
from sklearn.cluster import KMeans

# 클러스터 생성
k_cluster = KMeans(n_clusters=4) # 4개의 클러스터

# 모델 훈련
k_cluster.fit(points)

# 레이블 확인
# print(k_cluster.labels_) # 레이블
# print(np.shape(k_cluster.labels_)) # 차원(모양)
# print(np.unique(k_cluster.labels_)) # 유일한 레이블값

# 색상 딕셔너리
color_di = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'black'
}

# 그래프 그리기
# plt.figure(figsize=(10, 6))
# for cluster in range(4): # cluster : 클러스터 번호 (0~3)
#     cluster_sub = points[k_cluster.labels_==cluster]
#     plt.scatter(
#         cluster_sub[:, 0], # 첫번째 feature
#         cluster_sub[:, 1], # 두번째 feature
#         c = color_di[cluster], # 색상 딕셔너리에 정의한 클러스터별 색상
#         label = f'Cluster {cluster}' # 클러스터별 레이블
#     )
# plt.grid(True)
# plt.legend()
# plt.show()

# K-Means 원형 클러스터

# 라이브러리
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles # 원형의 무작위 데이터셋

# 무작위 데이터 생성
circle_points, circle_labels = make_circles(
    n_samples=150, # 샘플데이터의 수
    factor=0.5, # 안쪽 원과 바깥쪽 원의 크기 비율
    noise=0.01 # 노이즈, 값이 0에 가까울수록 노이즈가 적음
)

# 모델 생성
circle_kmeans = KMeans(n_clusters=2) # 클러스터 2개

# 모델 학습
circle_kmeans.fit(circle_points)

# 색상 딕셔너리
color_di = {0:'blue', 1:'red'}

# 스캐터
# for i in range(2):
#     cluster_sub = circle_points[circle_kmeans.labels_==i]
#     plt.scatter(
#         cluster_sub[:, 0],
#         cluster_sub[:, 1],
#         c = color_di[i],
#         label = f'Cluster_{i}'
#     )
# plt.figure(figsize=(10, 6))
# plt.legend()
# plt.grid(True)
# plt.show()


## DBSCAN : 밀도 기반 클러스터링 알고리즘

# 라이브러리 로딩
from sklearn.cluster import DBSCAN

# 임의 포인트 생성
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=200, random_state=42)
# 변환행렬
transformation = [
    [0.6, -0.6],
    [-0.3, 0.8]
]
# dot() : 배열과 행렬사이의 행렬곱셉을 수행하여
#         X배열의 모든 데이터 포인트에 대한 선형 변환
diag_points = np.dot(X, transformation)

# 반경, 최소인접포인트수
epsilon = 0.3
minPts = 3

# DBSCAN 생성
diag_dbscan = DBSCAN(
    eps = epsilon, # 반경
    min_samples = minPts # 최소인접포인트수
)

# 모델 훈련
diag_dbscan.fit(diag_points)

# 클러스터의 수, DBSCAN의 클러스터번호는 음수값을 포함하므로 +1 해줘야 함
# -1이 노이즈
n_cluster = max(diag_dbscan.labels_) + 1
#print(diag_dbscan.labels_)

# 스캐터
figure = plt.figure(figsize=(10, 6))
axes = figure.add_subplot(111)
color_di = {0:'red',1:'blue',2:'green',3:'black',4:'orange',5:'yellow', 6:'coral'}
for i in range(n_cluster):
    cluster_sub = diag_points[diag_dbscan.labels_==i]
    plt.scatter(
        cluster_sub[:, 0],
        cluster_sub[:, 1],
        c = color_di[i]
    )
axes.grid(True)
plt.show()





















































