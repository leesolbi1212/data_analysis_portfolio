# textmining2.py
# 기사댓글텍스트 분석

import pandas as pd

# 데이터프레임
df = pd.read_csv('assets/news_comment_BTS.csv')
#df.info()

# 한글이 아닌 모든 문자 제거
df['reply'] = df['reply'].str.replace('[^가-힣]', ' ', regex=True)
#print(df['reply'])

# Kkma 형태소 분석기를 통해 명사 추출
import konlpy
# Kkma 형태소 분석기
kkma = konlpy.tag.Kkma()
# reply 각각의 데이터에 kkma.nouns를 적용
nouns = df['reply'].apply(kkma.nouns)
#print(nouns)

# 리스트 분해
nouns = nouns.explode()
# print(nouns)

# 데이터프레임 생성
df_word = pd.DataFrame({'word': nouns})
#print(df_word)

# 글자 수 추가
df_word['count'] = df_word['word'].str.len()

# 두 글자 이상 단어 추출
df_word = df_word.query('count>=2')

# 단어별 빈도
df_word = df_word.groupby('word', as_index=False) \
                    .agg(n=('word', 'count')) \
                    .sort_values('n', ascending=False)

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
# sns.barplot(data=top20, x='word', y='n')
# plt.show()

# 워드클라우드

# 한글 폰트 설정
font = 'assets/DoHyeon-Regular.ttf'

# 데이터프레임을 딕셔너리로 변경
dic_word = df_word.set_index('word').to_dict()['n']
#print(dic_word)

#워드클라우드 그리기
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

# # 구름모양 워드클라우드 만들기
# import matplotlib.pyplot as plt
# from wordcloud import WordCloud
#
# # 이미지 로드
# from PIL import Image # 이미지 관련 라이브러리
# icon = Image.open('assets/cloud.png') # 이미지 메모리에 로딩
#
# # 마스크(mask) 생성 : 이미지 중에 덮을 부분
# import numpy as np
# # RGB모드로 icon이미지의 크기와 동일한 크기의 흰색 배경 이미지를 생성
# img = Image.new('RGB', icon.size, (255, 255, 255))
# # 흰색 배경위에 icon이미지를 덮어씀
# img.paste(icon, icon)
# # 이미지를 넘파이 배열로 변환
# img = np.array(img)
#
# # 워드클라우드 생성
# wc = WordCloud(
#     random_state = 1234,
#     font_path = font,
#     width = 400,
#     height = 400,
#     background_color = 'white',
#     mask = img, # 마스크로 사용할 넘파이 배열
#     colormap = 'inferno' # 컬러맵
# )
# img_wordcloud = wc.generate_from_frequencies(dic_word)
# plt.figure(figsize=(10, 10)) # 그래프 크기
# plt.axis('off') # 테두리선 없음
# plt.imshow(img_wordcloud) # 이미지 보여주기
# plt.show() # 워드클라우드 화면에 표시

























