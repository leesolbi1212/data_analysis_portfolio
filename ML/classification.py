# classification.py
# 분류

# 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn import datasets # scikit-learn 외부라이브러리

# 붓꽃 데이터셋
iris = datasets.load_iris()
#print(iris)
#print(type(iris))

# 데이터프레임 변환 후 확인
#df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
#df_iris.info()

# key 확인
#print(iris.keys())

# 데이터셋 정보
#print(iris['DESCR'])

# 특성(변수)명 출력
#print(iris.feature_names)
#print(iris['feature_names'])

# 특성(feature) 데이터 (대문자 X)
# = 독립 변수 (분류를 확정하기 위한 데이터)
#print(iris['data'])
features = iris['data']
#print(features[:5])

# 레이블(Label) 데이터 (소문자 y)
# = 종속 변수 (분류)
#print(iris['target'])

# 특성과 레이블을 데이터프레임으로 변환
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
#print(df.head())

# 컬럼명에서 (cm) 제거 (데이터프레임의 컬럼 재구성)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
#print(df.head())

# target변수 추가
df['target'] = iris['target']
#print(df.tail())

# 중복데이터가 있는지 확인
#print(df.duplicated().sum()) # 중복데이터의 합

# 몇 행과 중복된 데이터가 있는지 데이터프레임으로 출력
#print(df.loc[df.duplicated(keep=False)]) # 중복된 행을 모두 선택

# 중복된 데이터 제거
df = df.drop_duplicates()
#print(df.duplicated().sum()) # 중복데이터의 합

# 시각화 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 설정
plt.rc('font', family='Malgun Gothic')

# 캔버스 사이즈
plt.rcParams['figure.figsize'] = (12, 10) # 가로, 세로

# displot 그리기 : 데이터 분포 확인에 사용되는 그래프
# x:x축에 표시할 특성, kind:그래프 종류, hue:범례, data:데이터프레임
#sns.displot(x='petal_width', kind='kde', data=df)
#plt.show()
#sns.displot(x='sepal_width', kind='kde', hue='target', data=df)
#plt.show()

# pairplot : 데이터프레임의 각 변수들간의 쌍(pair)들의 관계 확인에 사용되는 그래프
# 데이터프레임, 범례, 그래프 높이, 그래프 종류
#sns.pairplot(df, hue='target', height=3.5, diag_kind='kde')
#plt.show()

# Train데이터셋, Test데이터셋 분할
# Train데이터셋 : 머신러닝에 사용되는 데이터셋
# Test데이터셋 : 머신러닝 후에 결과를 테스트하기 위한 데이터셋
from sklearn.model_selection import train_test_split

# 데이터셋
iris_data = iris.data

# 레이블
iris_label = iris.target

X_train, X_test, y_train, y_test \
    = train_test_split(
    iris_data, # 훈련데이터
          iris_label, # 특성명
          test_size = 0.2, # 분리비율 : 80%는 훈련용, 20%는 테스트용
          random_state = 7 # 데이터 분리할때 어떤 데이터들을 추출해서 분리할지를
                           # 랜덤하게 결정하기 위한 랜덤 시드값
                           # 랜덤 시드값이 같으면 항상 같은 분리를 함
    )
print(X_train.shape) # 데이터의 모양=차원, 2차원(120, 4):120행 4열
print(X_test.shape) # 2차원(30, 4): 30행 4열
print(y_train.shape) # 1차원(120, ) : 120개
print(y_test.shape) # 1차원(30, ) : 30개








































