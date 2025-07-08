# ex-clustering.py
# K-Means, DBSCAN 클러스터링 실습

# =====================================================
# Mall_Customers 데이터를 활용한 고객 클러스터링 실습
# - KMeans 클러스터링
# - DBSCAN 클러스터링
# =====================================================

# 필요한 라이브러리 불러오기
import pandas as pd                # 데이터프레임 처리
import matplotlib.pyplot as plt   # 시각화
from sklearn.cluster import KMeans, DBSCAN  # 클러스터링 알고리즘
from sklearn.preprocessing import StandardScaler  # 데이터 정규화
import warnings
warnings.filterwarnings(action='ignore')

# 데이터
df = pd.read_csv('assets/Mall_Customers.csv')
# df.info()

#데이터 항목:
#CustomerID	고객 고유 ID
#Gender	성별 (Male/Female)
#Age	나이
#Annual Income (k$)	연간 수입 (천 달러 단위)
#Spending Score (1–100)	지출 점수 (높을수록 많이 씀)

# 클러스터링에 사용할 변수
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 데이터 시각화
# plt.rc('font', family='Malgun Gothic')
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], s=50)
# plt.xlabel('Annual Income (k$) - 연 소득')
# plt.ylabel('Spending Score (1-100) - 소비 점수')
# plt.title('📌 고객 분포 시각화')
# plt.grid(True)
# plt.show()

# 실습1. 클러스터 개수를 5개로 설정하여 K-Means 클러스터링 후 시각화
# kmeans = KMeans(n_clusters=5, random_state=42)
#
# y_kmeans = kmeans.fit_predict(X)
#
# plt.rc('font', family='Malgun Gothic')
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=50)
# centers = kmeans.cluster_centers_ # 클러스터 중점
# plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centers')
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.title('K-Means 클러스터링 결과')
# plt.legend()
# plt.grid(True)
# plt.show()

# 실습2. 반경 0.5, 최소인접포인트수를 5로 설정하여 DBSCAN 클러스터링 후 시각화
# dbscan = DBSCAN(eps=0.5, min_samples=5)
#
# y_dbscan = dbscan.fit_predict(X)
#
# plt.rc('font', family='Malgun Gothic')
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='plasma', s=50)
# plt.xlabel('Annual Income (k$)')
# plt.ylabel('Spending Score (1-100)')
# plt.title('DBSCAN 클러스터링 결과')
# plt.grid(True)
# plt.show()





























