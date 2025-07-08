# imdb.py
# imdb 데이터셋

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import imdb
from wordcloud import WordCloud

# 폰트 설정 (한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'

# 1. 데이터 로딩 (상위 10,000개 단어만 사용)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

print(f"✅ 훈련 데이터 크기: {len(X_train)}")
print(f"✅ 테스트 데이터 크기: {len(X_test)}")

# 2. 라벨 분포 확인 (0 = 부정, 1 = 긍정)
unique, counts = np.unique(y_train, return_counts=True)
label_dist = dict(zip(unique, counts))
print("✅ 라벨 분포:", label_dist)

# 3. 리뷰 길이 통계
review_lengths = [len(review) for review in X_train]
print(f"✅ 평균 리뷰 길이: {np.mean(review_lengths):.2f}")
print(f"✅ 최대 리뷰 길이: {np.max(review_lengths)}")

# 4. 리뷰 길이 히스토그램
plt.figure(figsize=(10, 5))
plt.hist(review_lengths, bins=50, color='skyblue', edgecolor='black')
plt.title('리뷰 길이 분포')
plt.xlabel('리뷰 길이 (단어 수)')
plt.ylabel('리뷰 개수')
plt.grid(True)
plt.show()

# 5. 라벨 분포 시각화
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train)
plt.title('긍정 / 부정 리뷰 개수')
plt.xlabel('리뷰 라벨 (0=부정, 1=긍정)')
plt.ylabel('개수')
plt.xticks([0, 1], ['부정', '긍정'])
plt.show()

# 6. 단어 사전 불러오기
word_index = imdb.get_word_index()
index_word = {index + 3: word for word, index in word_index.items()}
index_word[0] = '<PAD>'
index_word[1] = '<START>'
index_word[2] = '<UNK>'
index_word[3] = '<UNUSED>'

# 7. 모든 리뷰 텍스트로 복원하여 단어 수집 (훈련 데이터 일부 샘플만 사용)
decoded_reviews = []
for i in range(1000):  # 상위 1000개만
    decoded = ' '.join([index_word.get(idx, '') for idx in X_train[i]])
    decoded_reviews.append(decoded)

all_words = ' '.join(decoded_reviews)

# 8. 워드클라우드 생성
wc = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("IMDB 리뷰에서 자주 등장하는 단어")
plt.show()
