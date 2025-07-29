import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU 사용 가능:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("⚠️ GPU를 찾지 못했습니다. CPU만 사용합니다.")
