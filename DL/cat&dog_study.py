import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array

# 1) 저장된 모델 불러오기
model = tf.keras.models.load_model('cat_dog_classifier.h5')
class_names = ['cat', 'dog']

# 2) 전처리 함수: 모델 내부 Rescaling 레이어가 있으므로 0–255 그대로 전달
def preprocess_image(img_path, target_size=(180, 180)):
    img = load_img(img_path, target_size=target_size)
    x   = img_to_array(img)            # (180,180,3), dtype=float32, 0~255
    x   = np.expand_dims(x, axis=0)    # (1,180,180,3)
    return x

# 3) 확률 비교 방식으로 예측 함수
def predict_image(img_path):
    x = preprocess_image(img_path)
    logit = model.predict(x)[0][0]         # 로짓 값
    prob_dog = tf.sigmoid(logit).numpy()   # dog일 확률
    prob_cat = 1.0 - prob_dog              # cat일 확률

    # 더 높은 확률 클래스로 결정
    if prob_cat > prob_dog:
        label, prob = 'cat', prob_cat
    else:
        label, prob = 'dog', prob_dog

    return {
        'label': label,
        'prob':  prob,
        'P(cat)': prob_cat,
        'P(dog)': prob_dog
    }
# 5) 테스트할 5개 이미지 경로
image_paths = [
    'assets/2.jpeg',
    'assets/12.jpeg',
    'assets/4.jpeg',
    'assets/1.jpg',
    'assets/3.jpg',
    'assets/7.jpg',
    'assets/9.jpg',
    'assets/11.jpg',
]

# 5) 예측 실행 및 출력
for img_path in image_paths:
    res = predict_image(img_path)
    print(f"{img_path} → 예측: {res['label']} ({res['prob']*100:.1f}%)")
    print(f"    P(cat)={res['P(cat)']*100:.1f}%, P(dog)={res['P(dog)']*100:.1f}%\n")

    # 이미지와 함께 시각화
    img = load_img(img_path)
    plt.imshow(img)
    plt.title(f"{res['label']} ({res['prob']*100:.1f}%)")
    plt.axis('off')
    plt.show()