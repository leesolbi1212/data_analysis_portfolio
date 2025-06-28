# datagraphexer2.py
# 데이터 정제 및 그래프 실습2 (그래프 하나에 통합)

# 라이브러리 임포트
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 데이터 로드
df = pd.read_csv("assets/economics_anormaly.csv")

# 고용률 이상치 처리
df["고용률"] = np.where(df["고용률"] > 100, np.nan, df["고용률"])

# 총고용자수와 고용률 결측치 제거
df = df.dropna(subset=["총고용자수", "고용률"])

# 실업자수 이상치 처리
df["실업자수"] = pd.to_numeric(df["실업자수"], errors="coerce")
q1 = df["실업자수"].quantile(0.25)
q3 = df["실업자수"].quantile(0.75)
iqr = q3 - q1
low = q1 - 1.5 * iqr
upp = q3 + 1.5 * iqr
df = df[(df["실업자수"] >= low) & (df["실업자수"] <= upp)]

# 조사년월 → datetime → 월 문자열로 변환
df["조사년월"] = pd.to_datetime(df["조사년월"], format="%Y-%m-%d")
df["조사년월_월"] = df["조사년월"].dt.to_period("M").astype(str)

# 그래프 설정
plt.rcParams.update({
    "font.size": 11,
    "font.family": "Malgun Gothic"
})

# 서브플롯 생성
fig, axs = plt.subplots(2, 2, figsize=(18, 10))

# 1. 물가지수 vs 고용률 산점도
sns.scatterplot(data=df, x="물가지수", y="고용률", ax=axs[0, 0])
axs[0, 0].set_title("물가지수별 고용률 산점도")

# 2. 조사년월별 고용률 막대그래프
sns.barplot(data=df, x="조사년월_월", y="고용률", ax=axs[0, 1])
axs[0, 1].set_title("조사년월별 고용률 막대그래프")
axs[0, 1].tick_params(axis='x', rotation=45)

# 3. 조사년월별 물가지수 라인그래프
sns.lineplot(data=df, x="조사년월_월", y="물가지수", ax=axs[1, 0])
axs[1, 0].set_title("조사년월별 물가지수 추이")
axs[1, 0].tick_params(axis='x', rotation=45)

# 4. 조사년월별 고용률 박스플롯
sns.boxplot(data=df, x="조사년월_월", y="고용률", ax=axs[1, 1])
axs[1, 1].set_title("조사년월별 고용률 분포 (Boxplot)")
axs[1, 1].tick_params(axis='x', rotation=45)

# 여백 자동 조정
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 상단 타이틀 공간 확보
plt.show()