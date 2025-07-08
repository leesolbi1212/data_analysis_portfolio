# textmining1.py
# 대통령 연설문 텍스트 분석 및 시각화

# 텍스트 로딩
moon = open('assets/speech_moon.txt', encoding='utf-8').read()
#print(moon)

# 한글이 아닌 모든 문자 제거
import re
# 인자 : 패턴문자열, 대체문자열, 전체문자열
moon = re.sub('[^가-힣]',' ', moon)
#print(moon)

# 명사 추출
# konlpy : 한글 형태소 분석 라이브러리, java설치 필요(jdk11)
# 1. https://jdk.java.net/java-se-ri/11
# 2. Windows/x64 Java Development Kit (sha256) 파일 다운로드
# 3. 다운로드 받은 파일 압축 해제 후 java-11 디렉토리명을 java11로 변경
# 4. java11디렉토리를 D:\ai2504\에 복사
# 5. 내컴퓨터 우클릭 > 속성 > 고급시스템설정 > 고급 탭 > 환경변수 > 시스템환경변수
# 6. 새로만들기 > 변수명:JAVA_HOME, 변수값: D:\ai2504\java11
# 7. path변수 편집 > D:\ai2504\java11\bin 추가
# 설치테스트 : CMD > java -version
import konlpy
hannanum = konlpy.tag.Hannanum()
nouns = hannanum.nouns(moon)
# print(nouns)

# 추출한 명사를 데이터프레임으로
import pandas as pd
df_word = pd.DataFrame({'word': nouns})
#print(df_word)
#df_word.info()

# 각 명사의 글자수 추가
df_word['count'] = df_word['word'].str.len()
#print(df_word)

# 글자수가 2 이상인 것 추출
df_word = df_word.query('count>=2')
df_word = df_word.sort_values('count', ascending=False)
#print(df_word)

# 단어별 빈도 구하기
df_word = df_word.groupby('word', as_index=False) \
                .agg(n=('word', 'count')) \
                .sort_values('n', ascending=False)
#print(df_word)

# 단어별 빈도 막대그래프용 데이터 20개 추출
top20 = df_word.head(20)

# 막대그래프 그리기
# import matplotlib.pyplot as plt
# import seaborn as sns
# # 그래프 설정
# plt.rcParams.update({
#     'font.family': 'Malgun Gothic', # 글자체
#     'figure.dpi': '120', # 해상도
#     'figure.figsize': [6.5, 6] # 그래프 크기
# })
# sns.barplot(data=top20, x='n', y='word')
# plt.show()

# 워드클라우드

# 한글 폰트 설정
font = 'assets/DoHyeon-Regular.ttf'

# 데이터프레임을 딕셔너리로 변경
dic_word = df_word.set_index('word').to_dict()['n']
#print(dic_word)

# 워드클라우드 그리기
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# # 워드클라우드 형태 설정
# wc = WordCloud(
#     random_state = 1234, # 워드클라우드의 모양을 랜덤하게 하는 난수
#     font_path = font, # 글자체
#     width = 400, # 넓이
#     height = 400, # 높이
#     background_color = 'white' # 배경색
# )
# # 워드클라우드 데이터 설정
# img_wordcloud = wc.generate_from_frequencies(dic_word)
# plt.figure(figsize=(10, 10)) # 그래프 크기
# plt.axis('off') # 테두리선 없음
# plt.imshow(img_wordcloud) # 이미지 보여주기
# plt.show() # 워드클라우드 화면에 표시

# 구름모양 워드클라우드 만들기
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 이미지 로드
from PIL import Image # 이미지 관련 라이브러리
icon = Image.open('assets/cloud.png') # 이미지 메모리에 로딩

# 마스크(mask) 생성 : 이미지 중에 덮을 부분
import numpy as np
# RGB모드로 icon이미지의 크기와 동일한 크기의 흰색 배경 이미지를 생성
img = Image.new('RGB', icon.size, (255, 255, 255))
# 흰색 배경위에 icon이미지를 덮어씀
img.paste(icon, icon)
# 이미지를 넘파이 배열로 변환
img = np.array(img)

# 워드클라우드 생성
wc = WordCloud(
    random_state = 1234,
    font_path = font,
    width = 400,
    height = 400,
    background_color = 'white',
    mask = img, # 마스크로 사용할 넘파이 배열
    colormap = 'inferno' # 컬러맵
)
img_wordcloud = wc.generate_from_frequencies(dic_word)
plt.figure(figsize=(10, 10)) # 그래프 크기
plt.axis('off') # 테두리선 없음
plt.imshow(img_wordcloud) # 이미지 보여주기
plt.show() # 워드클라우드 화면에 표시



















