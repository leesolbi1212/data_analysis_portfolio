# clustering_df.py
# 클러스터링을 위한 데이터 전처리

# 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings(action='ignore')

# rent.csv
data = pd.read_csv('assets/rent.csv')
#data.info()

# rent.csv 변수
'''
Posted On 게시된 날짜
BHK 침실(Bedroom), 홀(Hall), 주방(Kitchen)의 수
Rent 임대료
Size 크기
Floor 층 수
Area Type 주택이 속한 지역의 유형
Area Locality 지역의 위치
City 도시의 이름
Furnishing Status 주택 또는 아파트의 가구 유무 상태
Tenant Preferred 선호하는 임차인 유형
Bathroom 욕실 수
Point of Contact 문의할 담당자
'''

# 컬럼명 변경
new_column_name = {
  "Posted On": "Posted_On",
  "BHK": "BHK",
  "Rent": "Rent",
  "Size": "Size",
  "Floor": "Floor",
  "Area Type": "Area_Type",
  "Area Locality": "Area_Locality",
  "City": "City",
  "Furnishing Status": "Furnishing_Status",
  "Tenant Preferred": "Tenant_Preferred",
  "Bathroom": "Bathroom",
  "Point of Contact": "Point_of_Contact"
}
data.rename(columns = new_column_name, inplace=True)

# BHK값들을 오름차순으로 정렬
data['BHK'].sort_values()

# Rent 확인
#print(data['Rent'].value_counts())
#print(data['Rent'].sort_values())

# 아웃라이어 확인을 위한 박스플랏
# plt.figure(figsize=(8, 6))
# sns.boxplot(x=data['Rent'])
# plt.show()

# 아웃라이어 확인을 위한 스캐터
# plt.figure(figsize=(8, 6))
# plt.scatter(x=data.index, y=data['Rent'])
# plt.show()

# BHK와 Rent의 상관관계
# corr_BR = data['BHK'].corr(data['Rent'])
# print(f'BHK와 Rent의 상관관계 : {corr_BR:.2f}')

# BHK와 Rent 스캐터
# plt.scatter(data['BHK'], data['Rent'])
# plt.grid(True)
# plt.show()

# Size 확인
# print(data['Size'].value_counts())
# print(data['Size'].sort_values())

# Size displot
# sns.displot(data['Size'])
# plt.show()

# Size와 Rent 관계
# plt.scatter(data['Size'], data['Rent'])
# plt.show()

# 상관관계 확인

# 1. Rent와 BHK
#print(f"Rent와 BHK 상관관계 : {data['Rent'].corr(data['BHK'])}")

# 2. Rent와 Size
#print(f"Rent와 Size 상관관계 : {data['Rent'].corr(data['Size'])}")

# 3. Rent와 City
cities = data['City'].unique() # 유일한 값
#print(cities)
for city in cities:
    city_data = data[data['City']==city]
# 도시별 평균 임대료
city_mean = data.groupby('City')['Rent'].mean()
#print(city_mean)
# 도시명은 문자이므로 수치로 변환
data['City_mean'] = data.groupby('City')['Rent'].transform('mean')
#print(data['City_mean'])

# 상관관계 확인
#print(f"Rent와 City평균임대료의 상관관계 : {data['Rent'].corr(data['City_mean'])}")

# City, Rent그룹과 Rent 상관관계
rent_city = data.groupby('City')['Rent'].corr(data['Rent'])
#print(rent_city)

# 수치형 데이터들로 heatmap 그리기

# 수치형 변수만 선택
# numeric_data = data.select_dtypes(include=['int64', 'float64'])
# plt.figure(figsize=(10, 8))
# sns.heatmap(numeric_data.corr(), annot=True)
# plt.show()

# 지역별 임대료 분포 시각화
# plt.figure(figsize=(10, 6))
# sns.boxplot(x='City', y='Rent', data=data)
# plt.grid(True)
# plt.show()

# 도시별 평균임대료 내림차순
# avg_rent_city = data.groupby('City')['Rent'].mean() \
#     .sort_values(ascending=False)
# print(avg_rent_city)

# 날짜데이터 변환
data['Posted_On'] = pd.to_datetime(data['Posted_On'])
data['Year'] = data['Posted_On'].dt.year
data['Month'] = data['Posted_On'].dt.month
# print(data['Year'].value_counts())
# print(data['Month'].value_counts())

# 월별 평균임대료
avg_month_rent = data.groupby(['Year', 'Month'])['Rent'].mean()
#print(avg_month_rent)

# 월별 평균임대료 시각화
# plt.figure(figsize=(12, 6))
# avg_month_rent.plot(kind='line', marker='o')
# plt.grid(True)
# plt.show()

# 모델 선택
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 필요한 컬럼 선택
features = ['BHK', 'Size', 'Floor', 'Bathroom']
data1 = data[features + ['Rent']]
#print(data1)

# Floor의 문자열에서 숫자만 추출해서 float변환
data1['Floor'] = data1['Floor'].str.extract(r'(\d+)').astype(float)
print(data1['Floor'])

# 결측치 처리
data1 = data1.dropna() # 결측치가 있는 행 삭제

# 훈련/테스트 분리
X = data1[features]
y = data1['Rent']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 선형회귀 모델
lr = LinearRegression()

# 모델 훈련
lr.fit(X_train, y_train)

# 예측값
pred = lr.predict(X_test)

# MSE(평균제곱오차)
# 2542273917.555011 > 루트 씌우면 50,422
# 예측된 임대료와 실제임대료 사이에는 5만정도 오차
mse = mean_squared_error(y_test, pred)
#print(f'평균제곱오차(MSE) : {mse}')

# 실제값과 예측값 시각화
# plt.figure(figsize=(12, 6))
# plt.scatter(y_test, pred)
# plt.plot(
#     [min(y_test), max(y_test)],
#     [min(pred), max(pred)],
#     color = 'red',
#     linestyle='--'
# )
# plt.grid(True)
# plt.show()































