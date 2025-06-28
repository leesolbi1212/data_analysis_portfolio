# dataexer2.py
# 데이터분석 실습 2

# 필요한 라이브러리
import pandas as pd

# 아래 순서대로 실습을 해봅시다!
# 1. kaggle.com에서 Samsung Stocks CSV 획득
# kagglehub 외부라이브러리 필요
import kagglehub
# 데이터셋을 로컬저장하고 저장된 디렉토리를 반환
path = kagglehub.dataset_download("ranugadisansagamage/samsung-stocks")
print("Path to dataset files:", path)
df = pd.read_csv(f'{path}/Samsung.csv')
#print(df)

# 날짜 형식 변환
df['Date'] = pd.to_datetime(df['Date'])

# 연월 컬럼 생성
df['YearMonth'] = df['Date'].dt.to_period('M')

# 연도 컬럼 생성
df['Year'] = df['Date'].dt.year

# 2. 월별 최고가 및 평균 종가 출력

monthly_summary = df.groupby('YearMonth').agg(
    max_high = ('High', 'max'),
    mean_close = ('Close', 'mean')
)
print(monthly_summary)

# 3. 연도별 최저가 출력

yearly_low = df.groupby('Year').agg(
    min_low = ('Low', 'min')
)
print(yearly_low)








