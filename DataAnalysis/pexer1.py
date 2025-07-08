# pexer1.py
# 텍스트마이닝 실습 - gimi.txt (독립선언문 한글버젼)
# 텍스트 데이터 로딩
# 한글 추출
# 명사 추출
# 글자수 2개 이상인 단어 추출
# 단어별 빈도수 구하기
# 빈도수 상위 10개 막대그래프
# 워드클라우드
# 구름모양 워드클라우드

# 텍스트 로딩
gimi = open('assets/gimi.txt', encoding='utf-8').read()

# 한글이 아닌 모든 문자 제거
import re
gimi = re.sub('[^가-힣]',' ', gimi)

# 명사 추출
import konlpy
hannanum = konlpy.tag.Hannanum()
nouns = hannanum.nouns(gimi)

# 추출한 명사를 데이터프레임으로
import pandas as pd
df_word = pd.DataFrame({'word': nouns})

# 각 명사의 글자수 추가
df_word['count'] = df_word['word'].str.len()

# 글자수가 2 이상인 것 추출
df_word = df_word.query('count>=2')
df_word = df_word.sort_values('count', ascending=False)

# 단어별 빈도 구하기
df_word = df_word.groupby('word', as_index=False) \
                .agg(n=('word', 'count')) \
                .sort_values('n', ascending=False)

# 단어별 빈도 막대그래프용 데이터 10개 추출
top10 = df_word.head(10)

# 막대그래프 그리기
import matplotlib.pyplot as plt
import seaborn as sns
# 그래프 설정
plt.rcParams.update({
    'font.family': 'Malgun Gothic', # 글자체
    'figure.dpi': '120', # 해상도
    'figure.figsize': [6.5, 6] # 그래프 크기
})
sns.barplot(data=top10, x='n', y='word')
plt.show()

# 워드클라우드

# 한글 폰트 설정
font = 'assets/DoHyeon-Regular.ttf'

# 데이터프레임을 딕셔너리로 변경
dic_word = df_word.set_index('word').to_dict()['n']

# 워드클라우드 그리기
import matplotlib.pyplot as plt
from wordcloud import WordCloud
wc = WordCloud(
    random_state = 1234, # 워드클라우드의 모양을 랜덤하게 하는 난수
    font_path = font, # 글자체
    width = 400, # 넓이
    height = 400, # 높이
    background_color = 'white' # 배경색
)
img_wordcloud = wc.generate_from_frequencies(dic_word)
plt.figure(figsize=(10, 10)) # 그래프 크기
plt.axis('off') # 테두리선 없음
plt.imshow(img_wordcloud) # 이미지 보여주기
plt.show() # 워드클라우드 화면에 표시

# 구름모양 워드클라우드 만들기
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 이미지 로드
from PIL import Image # 이미지 관련 라이브러리
icon = Image.open('assets/cloud.png') # 이미지 메모리에 로딩

# 마스크(mask) 생성 : 이미지 중에 덮을 부분
import numpy as np
img = Image.new('RGB', icon.size, (255, 255, 255))
img.paste(icon, icon)
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




































