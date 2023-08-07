# 4import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

latent_dim = 32
num_frames = 100

# 构建变分自编码器模型
encoder_inputs = tf.keras.Input(shape=(784,))
x = layers.Dense(256, activation='relu')(encoder_inputs)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

# 采样潜在空间向量
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_var])

# 解码器模型
decoder_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(256, activation='relu')(decoder_inputs)
outputs = layers.Dense(784, activation='sigmoid')(x)

# 定义变分自编码器模型
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
decoder = tf.keras.Model(decoder_inputs, outputs, name='decoder')

# 定义变分自编码器的训练模型
vae = tf.keras.Model(encoder_inputs, decoder(encoder(encoder_inputs)[2]), name='vae')

# 加载已经训练好的变分自编码器权重
vae.load_weights('vae_weights.h5')

# 生成视频帧序列
video_frames = []
for _ in range(num_frames):
    # 采样潜在空间向量
    random_latent_vectors = tf.keras.backend.random_normal(shape=(1, latent_dim))
    # 使用解码器生成图像
    generated_images = decoder.predict(random_latent_vectors)
    # 将生成的图像添加到视频帧序列
    video_frames.append(generated_images.reshape((28, 28)) * 255)

# 保存视频帧序列为视频文件
out = cv2.VideoWriter('generated_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (28, 28), isColor=False)
for frame in video_frames:
    out.write(frame.astype(np.uint8))
out.release()
