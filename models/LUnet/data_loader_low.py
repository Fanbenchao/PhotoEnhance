import tensorflow as tf

image_feature_description = {
        'file_num': tf.io.FixedLenFeature([], tf.int64),
        'image_shape': tf.io.FixedLenFeature(3, tf.int64),
        'label': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
    }
def _parse_image_function(example_proto):
    return tf.io.parse_single_example(example_proto, image_feature_description)

def parse_image_train(image,label):
    image = tf.image.decode_jpeg(image,channels=3)
    label = tf.image.decode_jpeg(label,channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.image.convert_image_dtype(label,tf.float32)
    image_label = tf.concat([image,label],axis = -1)
    image_label = tf.image.random_flip_left_right(image_label)
    image_label = tf.image.random_crop(image_label, [512, 512,6])
    image_label = tf.image.resize(image_label,[256,256],method=tf.image.ResizeMethod.BICUBIC)
    image = image_label[:,:,:3]
    label = image_label[:,:,3:]
    return image, label
def load_and_preprocess_image_train(image_label):
    image = image_label['image']
    label = image_label['label']
    return parse_image_train(image,label)
def parse_image_test(image,label):
    image = tf.image.decode_jpeg(image,channels=3)
    label = tf.image.decode_jpeg(label,channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.image.convert_image_dtype(label,tf.float32)
    shape = tf.shape(image)
    height_offset = tf.compat.v1.to_int32((shape[0] - 512) / 2)
    width_offset = tf.compat.v1.to_int32((shape[1] - 512) / 2)
    image = tf.image.crop_to_bounding_box(image, height_offset, width_offset, 512, 512)
    image = tf.image.crop_to_bounding_box(label, height_offset, width_offset, 512, 512)
    # image = tf.image.resize(image,[1200,900],method=tf.image.ResizeMethod.BICUBIC)
    # label = tf.image.resize(label,[1200,900],method=tf.image.ResizeMethod.BICUBIC)
    return image, label
def load_and_preprocess_image_test(image_label):
    image = image_label['image']
    label = image_label['label']
    return parse_image_test(image,label)
def file_based_input_fn_builder(file_path,batch_size,is_training,buffer_size = 1000):
    dataset = tf.data.TFRecordDataset(file_path,buffer_size = 1000)
    dataset = dataset.map(_parse_image_function)
    if is_training:
        dataset = dataset.map(load_and_preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(load_and_preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset