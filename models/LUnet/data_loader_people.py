import tensorflow as tf
import numpy as np

def load(image_file,label_file):
    image = tf.io.read_file(image_file)
    label = tf.io.read_file(label_file)
    image = tf.image.decode_jpeg(image)
    label = tf.image.decode_jpeg(label)
    input_image = tf.cast(image, tf.float32)
    real_image = tf.cast(label, tf.float32)
    return input_image, real_image
def central_crop(input_image,real_image,height,width):
    shape = tf.shape(input_image)
    height_offset = tf.compat.v1.to_int32((shape[0] - height) / 2)
    width_offset = tf.compat.v1.to_int32((shape[1] - width) / 2)
    input_image = tf.image.crop_to_bounding_box(input_image, height_offset, width_offset,height, width)
    real_image = tf.image.crop_to_bounding_box(real_image, height_offset, width_offset,height, width)
    return input_image,real_image
def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width])
    real_image = tf.image.resize(real_image, [height, width])
    return input_image, real_image
def random_crop(input_image, real_image):
    stacked_image = tf.concat([input_image, real_image], axis=-1)
    cropped_image = tf.image.random_crop(stacked_image, size=[512, 512, 6])
    input_image,real_image = resize(cropped_image[:,:,:3], cropped_image[:,:,3:],256,256)
    return input_image, real_image
def normalize(input_image, real_image):
    input_image = input_image/255.0
    real_image = real_image/255.0
    return input_image, real_image
@tf.function()
def random_jitter(input_image, real_image):
    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop(input_image, real_image)
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image
def load_image_train(image_file,label_file):
    input_image, real_image = load(image_file,label_file)
    input_image = tf.image.central_crop(input_image, 0.8)
    real_image = tf.image.central_crop(real_image,0.8)
    scale = tf.keras.backend.random_uniform((1,), minval=0.6, maxval=1.0)[0]
    shape = tf.shape(input_image)
    input_image = tf.image.resize(input_image,[tf.compat.v1.to_int32(scale*tf.compat.v1.to_float(shape[0])),
                                               tf.compat.v1.to_int32(scale*tf.compat.v1.to_float(shape[1]))])
    real_image = tf.image.resize(real_image,[tf.compat.v1.to_int32(scale*tf.compat.v1.to_float(shape[0])),
                                             tf.compat.v1.to_int32(scale*tf.compat.v1.to_float(shape[1]))])
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image
def load_image_test(image_file,label_file):
    input_image, real_image = load(image_file,label_file)
    # input_image = tf.image.resize(input_image,[768,1024])
    # real_image = tf.image.resize(real_image, [768,1024])
    input_image, real_image = central_crop(input_image, real_image,512,512)
    input_image, real_image = normalize(input_image, real_image)
    return input_image, real_image
#1280,2048