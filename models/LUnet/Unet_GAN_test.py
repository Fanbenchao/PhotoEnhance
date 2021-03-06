import os
import time
import tensorflow as tf
import datetime
import numpy as np
from PIL import Image
import shutil
# from data_loader_people import load_image_train, load_image_test
from data_loader_low import file_based_input_fn_builder
from model import Generator, Discriminator
from loss import generator_loss, discriminator_loss
from glob import glob
from tqdm import tqdm

beta = 100
test_path = '/media/work/0005BBA40005E53D/lupin/dataset/SICE/test.tfrecords'
test_dataset = file_based_input_fn_builder(test_path, 1, is_training = False)

# test_path = '../dataset/crop_people/test.txt'
show_result = './prediction'
if not os.path.exists(show_result):
    os.mkdir(show_result)

# dirname = os.path.dirname(test_path)
# with open(test_path, 'r') as fid:
#     test_list = [l.strip() for l in fid.readlines()]
# test_img_files = [os.path.join(dirname, 'image', f) for f in test_list]
# test_label_files = [os.path.join(dirname, 'label', f) for f in test_list]
# test_img_list = tf.data.Dataset.list_files(test_img_files,shuffle= False)
# test_label_list = tf.data.Dataset.list_files(test_label_files,shuffle= False)
# test_dataset = tf.data.Dataset.zip((test_img_list,test_label_list))
# test_dataset = test_dataset.map(load_image_test)
# test_dataset = test_dataset.batch(1)
generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_best_dir = './Unet_attention_data_checkpoints/best'

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_best_dir))
test_psnr = 0
test_ssim = 0
for n, (input_image, target) in test_dataset.enumerate():
    prediction = generator(input_image, training=False)
    prediction = tf.clip_by_value(prediction,0,1)
    test_psnr += tf.reduce_mean(tf.image.psnr(target, prediction, max_val = 1.0))
    test_ssim += tf.reduce_mean(tf.image.ssim(target, prediction, max_val = 1.0))
    prediction = np.squeeze(prediction) * 255
    input_image = np.squeeze(input_image) * 255
    target = np.squeeze(target) * 255
    # prediction = Image.fromarray(np.uint8(prediction))
    # prediction.save("{0}/{1:06d}.jpg".format(show_result, n + 1))
    joint_im = np.concatenate((input_image, prediction, target), axis=1)
    joint_im = Image.fromarray(np.uint8(joint_im))
    joint_im.save("{0}/{1:06d}.jpg".format(show_result, n + 1))
n = tf.cast(n, dtype='float32')
test_psnr /= (n + 1)
test_ssim /= (n + 1)
print(test_psnr, test_ssim)
