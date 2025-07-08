# transformer.py
# TensorFlow로 구현한 Transformer

# 라이브러리
import tensorflow as tf
import numpy as np

# 1. 토큰 사전 정의 및 정수 인코딩
vocab = {"이란과": 1, "이스라엘의": 2, "전쟁은": 3, \
         "세계사적으로": 4, "미국이": 5, "주도하는": 6, \
         "힘의": 7, "분리를": 8, "함축한다": 9}
sentence = ["이란과", "이스라엘의", "전쟁은", "세계사적으로", "미국이", \
            "주도하는", "힘의", "분리를", "함축한다"]
token_ids = [vocab[token] for token in sentence]
x = tf.constant([token_ids], dtype=tf.int32)

# 2. 임베딩 레이어 : 실수 벡터로 변환
embedding = tf.keras.layers.Embedding(
    input_dim = 10, # 단어 집합 크기
    output_dim = 8 # 임베딩 차원
)
x_embed = embedding(x) # shape : (1, 4, 8) -> 행, 열, 임베딩차원

# 3. Self Attention
class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__() # 상위생성자 호출
        self.q_dense = tf.keras.layers.Dense(embed_dim) # Query 벡터 생성하는 Dense층
        self.k_dense = tf.keras.layers.Dense(embed_dim) # Key 벡터 생성하는 Dense층
        self.v_dense = tf.keras.layers.Dense(embed_dim) # Value 벡터 생성하는 Dense층
    def call(self, x):
        Q = self.q_dense(x) # Query 벡터 : (batch, seq_len, embed_dim)
        K = self.k_dense(x) # Key 벡터 : (batch, seq_len, embed_dim)
        V = self.v_dense(x) # Value 벡터 : (batch, seq_len, embed_dim)
        # 어텐션 스코어 : Q*K^T (T는 벡터의 전치연산, 즉 행과 열을 바꿈)
        # shape : (batch, seq_len, seq_len)
        scores = tf.matmul(Q, K, transpose_b=True)
        # 정규화를 위한 scaling (embed_dim의 제곱근으로 나눔)
        d_k = tf.cast(tf.shape(K)[-1], tf.float32)
        scores = scores / tf.math.sqrt(d_k)
        # Softmax를 적용해서 어텐션 가중치 행렬 생성
        # shape : (batch, seq_len, seq_len)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        # 어텐션 출력 계산 (가중합 : attention_weights * V)
        # shape : (batch, seq_len, embed_dim)
        output = tf.matmul(attention_weights, V)
        return output, attention_weights

# 4. Self Attention 레이어 인스턴스화
attention_layer = SelfAttention(embed_dim=8)

# 5. Self Attention 적용
# 문맥 반영된 벡터와 어텐션 행렬 반환
output, weights = attention_layer(x_embed)

# 6. 결과 출력
print("입력 문장 임베딩 (x_embed):\n", x_embed.numpy()) # 임베딩된 입력 벡터
print("어텐션 출력 (각 단어의 문맥 반영 표현):\n", output.numpy()) # self attention 결과
print("어텐션 가중치 (단어간 관계 스코어):\n", weights.numpy()) # attention weights 행렬

# 7. 어텐션 가중치 시각화
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc("font", family="Malgun Gothic")
plt.figure(figsize=(12, 10))
sns.heatmap(
    weights[0].numpy(),
    xticklabels = sentence,
    yticklabels = sentence,
    cmap = "YlGnBu",
    annot = True,
    fmt = ".4f",
    cbar = True
)
plt.title("Self Attention 가중치 행렬")
plt.xlabel("주목 대상 단어")
plt.ylabel("기준 단어")
plt.tight_layout()
plt.show()






































