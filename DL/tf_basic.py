# tf_basic.py
# 텐서플로우 기초

# 텐서플로우 라이브러리
import tensorflow as tf

# 버젼 확인
# print(f'Tensorflow 버젼 : {tf.__version__}')

# 상수 텐서

# 값:5, 차원:0, 데이터타입:4바이트정수
tensor_int = tf.constant(5, dtype=tf.int32)
# tf.Tensor(5, shape=(), dtype=int32)
# print(tensor_int)

# 값:[5,3], 차원:1, 데이터타입:4바이트정수
tensor1 = tf.constant([5, 3])
# tf.Tensor([5 3], shape=(2,), dtype=int32)
# print(tensor1)

# 값:[[1 2 3][4 5 6][7 8 9]], 차원:2, 데이터타입:4바이트정수
tensor2 = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
# tf.Tensor([[1 2 3][4 5 6][7 8 9]],shape=(3, 3),dtype=int32)
# print(tensor2)

# 변수 텐서
tensor3 = tf.Variable(5)
# <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=5>
# print(tensor3)
tensor3.assign(10) # 값을 10으로 변경
# <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=10>
# print(tensor3)

# 넘파이 배열을 텐서로 변환
import numpy as np
numpy_arr = np.array([1, 2, 3, 4]) # 리스트로 넘파이 배열 생성
# [1 2 3 4] int64 <class 'numpy.ndarray'>
# print(numpy_arr, numpy_arr.dtype, type(numpy_arr))
tensor = tf.convert_to_tensor(numpy_arr)
# tf.Tensor([1 2 3 4], shape=(4,), dtype=int64) <dtype: 'int64'> <class 'tensorflow.python.framework.ops.EagerTensor'>
# print(tensor, tensor.dtype, type(tensor))

# 리스트를 텐서로 변환
# 파이썬의 리스트는 모든 데이터타입 저장 가능
# 텐서는 정수와 실수만 저장 가능
li = [1, 3, 3.14, 7, 10]
# [1, 3, 3.14, 7, 10] <class 'list'>
# print(li, type(li))
tensor2 = tf.convert_to_tensor(li)
# tf.Tensor([ 1. 3. 3.14 7. 10.], shape=(5,), dtype=float32) <dtype: 'int64'> <class 'tensorflow.python.framework.ops.EagerTensor'>
# print(tensor2, tensor.dtype, type(tensor2))

# 텐서를 넘파이 배열로 변환
numpy_ar = tensor.numpy()
# [1 2 3 4] <class 'numpy.ndarray'>
# print(numpy_ar, type(numpy_ar))

# 상수텐서 2개 생성
tensor1 = tf.constant(5, dtype=tf.int32)

tensor2 = tf.constant(7, dtype=tf.int32)

# 텐서 덧셈
result = tf.add(tensor1, tensor2)
# tf.Tensor(12, shape=(), dtype=int32)
# print(result)

# 텐서 행렬곱셈
matrix1 = tf.constant([[1, 2], [3, 4]])
matrix2 = tf.constant([[5, 6], [7, 8]])
result1 = tf.matmul(matrix1, matrix2)
# tf.Tensor(
# [[19 22]
#  [43 50]], shape=(2, 2), dtype=int32) <class 'tensorflow.python.framework.ops.EagerTensor'>
# print(result1, type(result1))

# 텐서 슬라이싱 연산

# 2차원 텐서 생성
tensor3 = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
slice_tensor = tensor3[0:2, 1:3] # 0행~1행, 1열~2열
# tf.Tensor([[2 3] [5 6]], shape=(2, 2), dtype=int32)
# print(slice_tensor)


'''
  # 레이어 (Layer)
  - 입력층 : 활성화함수 만나기전의 데이터를 입력받는 층
  - 은닉층 : 활성화함수를 사용해서 데이터를 출력으로 보내는 층
  - 출력층 : 활성화함수의 결과를 출력데이터로 변환하는 층
  
  # 활성화 함수 (Activation Function)
  - 인공신경망에서 각 뉴런의 출력을 결정하는 함수, 입력을 출력으로 변환하는 함수
  - 뉴런의 입력을 받아 어떤 임계값을 기준으로 출력을 결정함
  - 비선형 데이터에 적용해서 복잡한 신경망들을 모델링하거나 다양한 문제를 해결 
  
  # ReLU (Recitified Linear Unit)
  - 은닉층(Hidden Layer)에서 주로 사용하는 활성화 함수
  - 양수가 입력되면 양수를 출력, 음수가 입력되면 0을 출력
  - y = max(0, x) : x입력, y출력
'''

# 상수 텐서
# tf.Tensor([-5.14  2.51 -4.14  0.05], shape=(4,), dtype=float32)
tensor4 = tf.constant([-5.14, 2.51, -4.14, 0.05])
print(tensor4)

# ReLU 적용 결과
relu = tf.nn.relu(tensor4)
# 활성화함수 ReLU 적용 결과 : [0.   2.51 0.   0.05]
print(f'활성화함수 ReLU 적용 결과 : {relu.numpy()}')

# 텐서 모양 변경
tensor5 = tf.constant([1,2,3,4,5,6]) # 1차원
tensor_re = tf.reshape(tensor5, (2,3)) # 2차원
# tf.Tensor([[1 2 3] [4 5 6]], shape=(2, 3), dtype=int32)
print(tensor_re)
tensor_re2 = tf.reshape(tensor5, (1,6)) # 2차원
# tf.Tensor([[1 2 3 4 5 6]], shape=(1, 6), dtype=int32)
print(tensor_re2)

# 모든 요소가 0인 텐서
zeros_tf = tf.zeros((5,2)) # 5행 2열인데 모든 요소가 0
# [[0. 0.] [0. 0.] [0. 0.] [0. 0.] [0. 0.]]
print(zeros_tf.numpy())

# 모든 요소가 1인 텐서
ones_tf = tf.ones((4,3)) # 4행 3열인데 모든 요소가 1
# [[1. 1. 1.] [1. 1. 1.] [1. 1. 1.] [1. 1. 1.]]
print(ones_tf.numpy())

# 모든 요소를 특정값으로 하는 텐서
fill_tf = tf.fill((3,2), 10) # 3행 2열인데 모든 요소가 10
# tf.Tensor([[10 10] [10 10] [10 10]], shape=(3, 2), dtype=int32)
print(fill_tf)

# 정규화된 랜덤값으로 모든 요소를 채우는 텐서
tensor = tf.random.normal(
    (3, 4), # shape, 3행 4열
    mean = 0.0, # 평균
    stddev = 1.0, # 표준편차
    dtype = tf.float32, # 요소의 데이터타입은 4바이트 실수
    seed = 100 # 랜덤 시드값
)
# tf.Tensor(
# [[ 0.08766998 -0.26517144 -0.99532986  0.5877223 ]
#  [-1.0208673   0.40396553  0.4916224  -0.64681983]
#  [-0.12439691  0.28154397  1.0645627   2.2246978 ]], shape=(3, 4), dtype=float32)
print(tensor) # 평균:0.174099945, 표준편차:0.870493814454342
tensor_int = tf.cast(tensor, dtype=tf.int32) # 정수로 타입 변환
# tf.Tensor([[ 0  0  0  0] [-1  0  0  0] [ 0  0  1  2]], shape=(3, 4), dtype=int32)
print(tensor_int)
