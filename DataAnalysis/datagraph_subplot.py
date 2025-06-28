# datagraph_subplot.py
# 그래프내의 서브그래프

# 라이브러리 임포트
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 데이터프레임
df = pd.read_csv('assets/mpg.csv')

# 시각화 스타일 설정
sns.set(style='whitegrid')

# 그래프 크기 설정
plt.figure(figsize=(14, 8))

# 한글 폰트 설정
plt.rcParams.update({'font.family': 'Malgun Gothic'})

# 박스플랏 : 제조사별 도시연비
# subplot(가로개수, 세로개수, 순서)
# 2행 2열 중 첫번째 위치 (가로>세로>다음줄>가로>세로)
plt.subplot(2, 2, 1)
sns.boxplot(data=df, x='manufacturer', y='cty')
plt.xticks(rotation=90) # X축에 표시되는 틱(텍스트) 기울기
plt.title('제조사별 도시 연비') # 그래프 제목

# 스캐터플랏 : 배기량 vs 고속도로연비
plt.subplot(2, 2, 2)
sns.scatterplot(data=df, x='displ', y='hwy', hue='cyl', palette='viridis')
plt.title('배기량과 고속도로연비 관계')

# 바플랏 : 제조사별 평균 고속도로연비
plt.subplot(2, 2, 3)
# reset_index를 사용하여 DataFrame화 해야 함
mean_hwy = df.groupby('manufacturer')['hwy'].mean() \
    .sort_values(ascending=False).reset_index()
sns.barplot(data=mean_hwy, x='manufacturer', y='hwy')
plt.xticks(rotation=90)
plt.title('제조사별 평균 고속도로연비')

# 라인플랏 : 연도별 평균 도시연비
plt.subplot(2, 2, 4)
mean_cty_by_year = df.groupby('year')['cty'].mean().reset_index()
sns.lineplot(data=mean_cty_by_year, x='year', y='cty', marker='o')
plt.title('연도별 평균 도시연비')

# 내부 그래프들의 크기에 맞게 전체 그래프의 크기를 조절
plt.tight_layout()

# 그래프 화면에 표시
plt.show()































