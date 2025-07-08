# linearregression.py
# 선형회귀 : 시간별로 축적된 데이터들을 학습시켜서 미래 데이터를 예측

# 라이브러리 로딩
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트
plt.rc('font', family='Malgun Gothic')

# 데이터프레임
df = pd.read_csv('assets/melon.csv')
#df.info()

# 가수별 빈도
singer_counts = df['가수명'].value_counts()
#print(singer_counts)

# 가수별 빈도 시각화 : 막대그래프
# plt.figure(figsize=(10, 6))
# plt.bar(singer_counts.index, singer_counts.values)
# plt.xlabel('가수명')
# plt.ylabel('노래개수')
# plt.title('가수별 노래개수 분포')
# plt.xticks(rotation=45, ha='right') # 글자 기울임 각도, 정렬
# plt.tight_layout() # 내용물에 맞게 그래프 축소
# plt.show()

# 데이터프레임
df1 = pd.read_csv('assets/channel.csv')
#df1.info()

# 첫번째 컬럼명을 '연도'로 변경
df1 = df1.rename(columns={'Unnamed: 0': '연도'})
#df1.info()

# 그래프 실습1 : 연도별 각 채널의 시청률 변화 라인그래프
# plt.figure(figsize = (12, 6))
# for channel in df1.columns[1:]:
#     plt.plot(df1["연도"], df1[channel], label = channel)
# plt.xlabel("연도")
# plt.ylabel("시청률")
# plt.title("연도별 각 채널의 시청률 변화")
# plt.legend(loc = "upper right") # 범례 위치
# plt.grid(True) # 그리드 생성여부
# plt.show()

# 그래프 실습2 : 연도별 각 채널의 시청률 평균 막대그래프
# avg = df1.groupby("연도").mean() # 연도별 평균
# avg.plot(kind = "bar", figsize = (12, 6))
# plt.xlabel("연도")
# plt.ylabel("평균 시청률")
# plt.title("연도별 각 채널의 시청률 평균")
# plt.legend(title = '채널')
# plt.xticks(rotation = 0)
# plt.grid(axis = "y")
# plt.show()

# 선형 회귀 (Linear Regression)
# 종속변수 y와 한개 이상의 독립변수 X와의 선형 상관관계를
# 모델링하는 회귀분석 기법

# 라이브러리 임포트
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 랜덤데이터 시드
np.random.seed(0)

# 독립변수
# np.random.rand : 0.0이상 1.0미만의 임의의 실수
X = np.random.rand(100, 1)
#print(X)

# 종속변수 : X + 1 + 잡음(노이즈)
# np.random.randn : 평균이 0이고 표준편차가 1인 정규분포 값
y = X + 1 + 0.1 * np.random.randn(100, 1)
#print(y)

# 선형회귀모델 생성
model = LinearRegression()

# 모델 학습
model.fit(X, y) # 모델.fit(독립변수, 종속변수)

# 학습데이터를 산점도로 시각화
plt.scatter(X, y, label='학습데이터')

# 회귀선 데이터 생성: 0 ~ 1 구간의 X값
X_line = np.linspace(0, 1, 100).reshape(-1, 1)  # 0~1 사이 100개 점
y_line = model.predict(X_line)  # 해당 점들의 예측값

# 회귀선 시각화
plt.plot(X_line, y_line, color='red', label='회귀선')  # 선형 회귀선

# 개별 예측 예시
X_test = np.array([[0.5]])  # 예측을 원하는 입력값
y_pred = model.predict(X_test)  # 예측값
plt.scatter(X_test, y_pred, color='green', s=100, label='예측값(0.5)', marker='x')  # 예측점을 X로 표시

# 라벨 및 범례
plt.xlabel('X')
plt.ylabel('y')
plt.title('선형 회귀 시각화')
plt.legend()
plt.grid(True)
plt.show()



































