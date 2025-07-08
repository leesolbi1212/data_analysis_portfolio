# titanic_dataset.py
# 타이타닉 데이터셋

# 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 경고메세지 무시
import warnings
warnings.filterwarnings(action='ignore')

# train데이터셋
titanic_df = pd.read_csv('assets/train.csv')
#titanic_df.info()

# 타이타닉 train.csv 파일의 컬럼명
#PassengerId	탑승자 데이터 일련번호
#Survived	생존여부(0: 사망, 1: 생존)
#Pclass		티켓선실등급(1: 일등석, 2: 이등석, 3: 삼등석)
#Name		탑승자 이름
#Sex		탑승자 성별
#Age		탑승자 나이
#SibSp		같이 탑승한 형제, 자매 or 배우자의 인원 수
#Parch		같이 탑승한 부모님 or 어린이 인원 수
#Ticket		티켓번호
#Fare		요금
#Cabin		선실번호
#Embarked	중간정착항구(C : Cherbourg, Q : Queenstown, S = Sothampton)

# 컬럼별 null값이 몇개 있는지 확인
#print(titanic_df.isnull().sum())

# 나이는 평균값으로 채움
titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
#print(titanic_df.isnull().sum())

# 선실과 정착지는 N으로 채움
titanic_df['Cabin'].fillna('N', inplace=True)
titanic_df['Embarked'].fillna('N', inplace=True)
#print(titanic_df.isnull().sum())

# 출력결과를 모두 볼 수 있게 설정
pd.set_option('display.max_rows', 900)
#print(titanic_df)

# 중복값 확인
#print(titanic_df[titanic_df.duplicated()])

# 성별 승객수 확인
# print(titanic_df['Sex'].value_counts())

# 선실별 승객수 확인
#print(titanic_df['Cabin'].value_counts())

# 정착지별 승객수
#print(titanic_df['Embarked'].value_counts())

# 선실의 첫번째 알파벳만 추출 후 개수 확인
#print(titanic_df['Cabin'].str[:1].value_counts())

# 성별에 따른 생존자 수
# print(titanic_df.groupby('Sex')['Survived'].sum())

# 그래프 그리기

# 성별과 생존자수를 기준으로 바그래프 그리기
# ss = titanic_df.groupby(['Sex', 'Survived']).size().unstack()
# ss.plot(kind='bar')
# plt.show()
# sns.countplot(x='Sex', data=titanic_df, hue='Survived')
# plt.show()

# 나이별 카테고리
def category(age):
    re = ''
    if age <= -1:
        re = 'Unknown'
    elif age <=5:
        re = 'baby'
    elif age <=12:
        re = 'child'
    elif age <=19:
        re = 'teenager'
    elif age <=25:
        re = 'student'
    elif age <=35:
        re = 'young adult'
    elif age <=80:
        re = 'adult'
    else:
        re = 'elderly'
    return re

# 나이 카테고리 그래프
plt.figure(figsize=(10, 6))
# group_name = ['Unknown', 'baby', 'child', 'teenager', 'student' \
#               , 'young adult', 'adult', 'elderly']
# Age값들을 하나씩 꺼내서 category함수에 전달
# titanic_df['Age_Cate'] = titanic_df['Age'].apply(category)
# print(titanic_df['Age_Cate'])
# sns.barplot(
#     x='Age_Cate',
#     y='Survived',
#     hue='Sex',
#     data=titanic_df,
#     order=group_name # 그래프 표시 순서
# )
#plt.show()

# 탑승 요금 분포 그래프
# sns.histplot(
#     data=titanic_df,
#     x='Fare',
#     bins=30, # 구간
#     kde=True # 부드러운 곡선 표시 여부
# )
# plt.show()


## 인코딩

# 레이블 인코딩
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# 성별을 레이블 인코딩
titanic_df['Sex_LabelEncoder'] \
    = label_encoder.fit_transform(titanic_df['Sex'])
#print(label_encoder.classes_)
#print(titanic_df['Sex_LabelEncoder'])

# 레이블인코딩용 함수
def encode_features(df, features):
    for i in features:
        le = LabelEncoder()
        le = le.fit(df[i])
        df[i] = le.transform(df[i])
    return df
print(encode_features(titanic_df, ['Sex', 'Cabin', 'Embarked']))























