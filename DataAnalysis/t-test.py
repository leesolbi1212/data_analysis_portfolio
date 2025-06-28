# t-test.py
# t-test 검정

# 검정 : 통계적 가설을 검토해서 채택할지 기각할지를 판단하는 과정
# 검증 : 모형/시스템/이론 등이 올바른지 확인하는 과정

# t-test 검정 (t검정)
# 두 변수(또는 두 데이터군)의 "평균"을 기준으로 유사도를 검증하는 통계분석 기법
# scipy 외부 라이브러리 사용

# 데이터
import pandas as pd
from matplotlib.lines import lineStyles

mpg = pd.read_csv('assets/mpg.csv')

# 기술 통계 분석 : 기존 데이터를 활용한 통계 분석
# 카테고리가 compact이거나 suv, 카테고리별로 그룹핑
mpg_agg = mpg.query("category in ['compact', 'suv']") \
            .groupby('category', as_index=False) \
            .agg(
                n=('category', 'count'), # 카테고리별 개수
                mean=('cty', 'mean') # 도시연비 평균
            )
# print(mpg_agg)

# category별 cty 분리
compact = mpg.query("category=='compact'")['cty']
suv = mpg.query("category=='suv'")['cty']
# print(compact, suv)

# compact, suv t-test 검정
from scipy import stats
# equal_var=True	"두 그룹이 점수 퍼짐(분산)이 비슷하다고 믿고 검사하는 것"
# equal_var=False	"두 그룹이 점수 퍼짐이 다를 수도 있다고 생각해서 더 신중하게 검사하는 것"
# pvalue=np.float64(2.3909550904711282e-21) => 2.39... * 10의 -21승
# pvalue가 0.05보다 작음 => 유의미하다 => 우연히 차이가 발생할 확률이 적다
# => 도시연비는 compact, suv 이냐에 따라서 차이가 있다
# => compact, suv에 따라 도시연비에 영향을 준다
# print(stats.ttest_ind(compact, suv, equal_var=True))

# 실습 : 일반휘발유(fl:r)와 고급휘발유(fl:p)의 도시연비(cty) t-test검정 해보기!
mpg_fl = mpg.query("fl in ['r', 'p']") \
            .groupby('fl', as_index=False) \
            .agg(
                n = ('fl', 'count'),
                mean = ('cty', 'mean')
            )
regular = mpg.query("fl=='r'")['cty'] # 일반휘발유
premium = mpg.query("fl=='p'")['cty'] # 고급휘발유

print(mpg_fl)

# pvalue=np.float64(0.28752051088667036) > 0.05
# 유의미하지 않다 => 우연히 발생할 확률이 높다
# => 도시연비는 휘발유가 일반이냐 고급이냐에 영향을 크게 받지 않는다
print(stats.ttest_ind(regular, premium, equal_var=True))

# 필수 라이브러리 불러오기
import pandas as pd                     # 데이터 처리용 라이브러리
import seaborn as sns                   # 시각화 라이브러리 seaborn
import matplotlib.pyplot as plt         # 시각화 라이브러리 matplotlib

# 한글폰트 패치 (한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 사용자는 Malgun Gothic
plt.rcParams['axes.unicode_minus'] = False     # 음수(-)기호 깨짐 방지

# 데이터 불러오기 (자동차 연비 데이터)
mpg = pd.read_csv('assets/mpg.csv')

# 'compact', 'suv' 차량 데이터만 추출하기
mpg_compact_suv = mpg.query("category in ['compact', 'suv']")

# 그래프 크기 지정
plt.figure(figsize=(8, 6))

# 박스 플롯(boxplot) 생성
sns.boxplot(
    data=mpg_compact_suv,     # 사용할 데이터
    x='category',             # x축 (차량 종류)
    y='cty',                  # y축 (도시 연비)
    hue='category',           # 카테고리별 색 구분
    palette='Set2',           # 색상 팔레트 지정
    legend=False              # 범례 표시 안함 (중복 방지)
)

# 데이터 분포 표현을 위한 스웜 플롯(swarmplot) 추가
sns.swarmplot(
    data=mpg_compact_suv,     # 사용할 데이터
    x='category',             # x축 (차량 종류)
    y='cty',                  # y축 (도시 연비)
    color='.25'               # 점 색상 (어두운 회색)
)

# 그래프 제목 추가 (한글 지원 확인)
plt.title('Compact vs SUV 도시연비 비교 (t-test)', fontsize=16)

# 축 이름 지정 (한글 표시)
plt.xlabel('차량 종류(Category)', fontsize=14)
plt.ylabel('도시 연비(cty)', fontsize=14)

# Y축 눈금선 추가 (데이터 파악 용이하게 함)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 최종 그래프 출력
plt.show()












