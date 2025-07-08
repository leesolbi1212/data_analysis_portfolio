# economicsexer.py
# 데이터 정제 및 그래프 실습1

# 라이브러리 임포트
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 데이터 로드
df = pd.read_csv("assets/economics_anormaly.csv")
print(df.head())

# 1. 고용률 이상치를 NaN 처리
df["고용률"] = np.where(df["고용률"] > 100, np.nan, df["고용률"])

# 2. 총고용자수와 고용률 결측치 제거
df = df.dropna(subset=["총고용자수", "고용률"])

# 3. 실업자수 이상치 제거
# 실업자수 컬럼에서 문자형으로 되어 있는 값이 있는지 확인하고 제거
# errors="coerce" : 숫자로 변환할 수 없는 경우 NaN으로 처리
df["실업자수"] = pd.to_numeric(df["실업자수"], errors="coerce")
q1 = df["실업자수"].quantile(0.25)
q3 = df["실업자수"].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5 * iqr
upp = q3 + 1.5 * iqr
df = df[(df["실업자수"] >= low) & (df["실업자수"] <= upp)]

# 4. 조사년월 문자열 → datetime → 월 단위 문자열로 변환
df["조사년월"] = pd.to_datetime(df["조사년월"], format="%Y-%m-%d")
df["조사년월_월"] = df["조사년월"].dt.to_period("M").astype(str)

# 그래프 설정
plt.rcParams.update({
    "figure.dpi": 100,
    "figure.figsize": [14, 6],
    "font.size": 12,
    "font.family": "Malgun Gothic"
})

# 1. 물가지수 vs 고용률 산점도
sns.scatterplot(data=df, x="물가지수", y="고용률")
plt.title("물가지수별 고용률 산점도")
plt.tight_layout()
plt.show()

# 2. 조사년월별 고용률 막대그래프
sns.barplot(data=df, x="조사년월_월", y="고용률")
plt.xticks(rotation=45)
plt.title("조사년월별 고용률 막대그래프")
plt.tight_layout()
plt.show()

# 3. 조사년월별 물가지수 라인그래프
sns.lineplot(data=df, x="조사년월_월", y="물가지수")
plt.xticks(rotation=45)
plt.title("조사년월별 물가지수 추이")
plt.tight_layout()
plt.show()

# 4. 조사년월별 고용률 박스플롯
sns.boxplot(data=df, x="조사년월_월", y="고용률")
plt.xticks(rotation=45)
plt.title("조사년월별 고용률 분포 (Boxplot)")
plt.tight_layout()
plt.show()
