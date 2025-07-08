# encoding.py
# 인코딩

# 인코딩(Encoding) : 머신이 연산을 하기 위해 데이터를 숫자화하는 작업

'''
인코딩 방식
1. Label Encoding : 데이터에 순서를 부여해서 정수로 변환, sklearn.preprocessing
2. One-Hot Encoding : 데이터를 이진 벡터로 표현 (0, 1), pandas, sklearn,OneHotEncoder
3. Ordinal Encoding : 지정한 순서대로 정수로 변환, sklearn.preprocessing
4. Binary Encoding : 정수 > 2진수로 변환 > 벡터로 변환, category_encoders
5. Target Encoding : 평균 타겟값으로 인코딩, category_encoders
'''
import pandas as pd
# Label Encoding
# 문자를 비교해서 0부터 숫자로 변환
# 장점 : 간단하고 빠름
# 단점 : 변환된 숫자간의 순서에 의미가 없음
# from sklearn.preprocessing import LabelEncoder
# import pandas as pd
# df = pd.DataFrame({'Color':['Red','Blue','Green','Blue','Red']})
# le = LabelEncoder()
# df['Colore_encoded'] = le.fit_transform(df['Color'])
# print(df)

# One-Hot Encoding
# 하나만 True(1), 나머지는 False(0)
# 장점 : 순서 정보 제거, 범주간 간섭 없음
# 단점 : 차원이 매우 커질 수 있음
# import pandas as pd
# df = pd.DataFrame({'Color':['Red','Green','Blue']})
# one_hot = pd.get_dummies(df['Color'], prefix='Color')
# print(one_hot)

# Ordinal Encoding
# 순서가 있는 범주에 수를 부여
# 서열이 중요한 경우 사용
# from sklearn.preprocessing import OrdinalEncoder
# df = pd.DataFrame({'Grade':['Low','Medium','High','Medium']})
# encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
# df['Grade_encoded'] = encoder.fit_transform(df[['Grade']])
# print(df)

# Binary Encoding
# category-encoders 외부라이브러리 설치

# Label Encoding 후에 2진수로 변환 후
# Busan:0(000), Daegu:1(001), Incheon:2(010), Seoul:3(011)
# Busan : 000 > City_0(0), City_1(0), City_2(0)
#         맨 우측비트를 City_2에 할당 (원래의 binary encoding)

# category_encoders의 binary encoding은
# 맨 우측비트를 City_0에 할당, 왼쪽비트들은 City_1, City_2
# Seoul 3 110(이진수 역순)   City_0(0), City_1(1), City_2(1)
# Busan 1 100(이진수 역순)   City_0(0), City_1(0), City_2(1)
# Incheon 2 010(이진수 역순) City_0(0), City_1(1), City_2(0)
# Seoul 3 110(이진수 역순)   City_0(0), City_1(1), City_2(1)
# Daegu 4 001(이진수 역순)   City_0(1), City_1(0), City_2(0)

import pandas as pd
import category_encoders as ce
df = pd.DataFrame({'City':['Seoul','Busan','Incheon','Seoul','Daegu']})
encoder = ce.BinaryEncoder(cols=['City'])
df_encoded = encoder.fit_transform(df)
print(df_encoded)

# Target Encoding
# 범주형 값을 해당 카테고리의 평균 타겟값으로 변환
# A : 100, 150 => (100+150)/2 = 125.0
# B : 200, 180 => (200+180)/2 = 190.0
# C : 160 => 160/1 = 160.0
# 전체 평균과 그룹 평균을 가중합 (smoothing 사용)
# 전체 평균 = 790/5 = 158.0
# A 합계 = 250.0
# 250+158/3 = 136.0 + smoothing값
# import pandas as pd
# import category_encoders as ce
# df = pd.DataFrame({
#     'Team': ['A', 'B', 'A', 'B', 'C'],
#     'Score': [100, 200, 150, 180, 160]
# })
# encoder = ce.TargetEncoder(cols=['Team'])
# df['Team_encoded'] = encoder.fit_transform(df['Team'], df['Score'])
# print(df)
































