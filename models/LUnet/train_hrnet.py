import os
import time
import tensorflow as tf
import datetime
import numpy as np
from PIL import Image
import shutil
from data_loader_people import load_image_train, load_image_test
from HRnet import HRnet
from loss import L2_loss
from glob import glob
from tqdm import tqdm

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

buffer_size = 100
train_batch_size = 2
test_batch_size = 1
train_path = '../dataset/crop_people/train.txt'
test_path = '../dataset/crop_people/test.txt'
dirname = os.path.dirname(train_path)
with open(train_path, 'r') as fid:
    train_list = [l.strip() for l in fid.readlines()]
with open(test_path, 'r') as fid:
    test_list = [l.strip() for l in fid.readlines()]
train_img_files = [os.path.join(dirname, 'image', f) for f in train_list]
train_label_files = [os.path.join(dirname, 'label', f) for f in train_list]
test_img_files = [os.path.join(dirname, 'image', f) for f in test_list]
test_label_files = [os.path.join(dirname, 'label', f) for f in test_list]

train_img_list = tf.data.Dataset.list_files(train_img_files, shuffle=False)
train_label_list = tf.data.Dataset.list_files(train_label_files, shuffle=False)
train_dataset = tf.data.Dataset.zip((train_img_list, train_label_list))
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size)
train_dataset = train_dataset.batch(train_batch_size)

test_img_list = tf.data.Dataset.list_files(test_img_files, shuffle=False)
test_label_list = tf.data.Dataset.list_files(test_label_files, shuffle=False)
test_dataset = tf.data.Dataset.zip((test_img_list, test_label_list))
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(test_batch_size)

model = HRnet()

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4)
checkpoint_dir = './HRnet_checkpoint'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint_best = os.path.join(checkpoint_dir, 'best')
if not os.path.exists(checkpoint_best):
    os.mkdir(checkpoint_best)
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)

save_dir1 = '/'.join([checkpoint_dir,'test_result'])
save_dir2 = '/'.join([checkpoint_dir,'best_result'])
if not os.path.exists(save_dir1):
    os.mkdir(save_dir1)
if not os.path.exists(save_dir2):
    os.mkdir(save_dir2)
log_dir = '/'.join([checkpoint_dir,'log'])

summary_writer = tf.summary.create_file_writer(log_dir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape:
        prediction = model(input_image, training=True)
        loss = L2_loss(target,prediction)

    model_gradients = gen_tape.gradient(loss,
                                            model.trainable_variables)

    optimizer.apply_gradients(zip(model_gradients,
                                            model.trainable_variables))

    return loss

def fit(train_ds, end_epochs, test_ds):
    best_result_list = glob(save_dir2 + '/*')
    status = checkpoint.restore(manager.latest_checkpoint)
    if best_result_list:
        best_result_list = sorted(best_result_list, key=lambda x: float(x.split('/')[-1].split('_')[1]))
        best_epoch = [int(epoch.split('/')[-1].split('_')[0]) for epoch in best_result_list]
        best_psnr = [float(epoch.split('/')[-1].split('_')[1]) for epoch in best_result_list]
        start_epochs = os.listdir(save_dir1)
        start_epochs = sorted(start_epochs,key = lambda x: int(x))
        start_epochs = int(start_epochs[-1])
        print('Restoration Model from the lastest checkpoint!!')
    else:
        best_epoch = [0]
        best_psnr = [0]
        start_epochs = 0
    for epoch in range(start_epochs, end_epochs):
        start = time.time()
        print("Epoch: ", epoch + 1)

        psnr, loss = 0,0
        # Train
        for n, (input_image, target) in tqdm(train_ds.enumerate()):
           loss += train_step(input_image,target)
        n = tf.cast(n, dtype='float32')
        loss = loss/(n+1)
        with summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=epoch)
        print('loss: {:4.4f}'.format(loss))

        if (epoch + 1) % 50 == 0:
            test_psnr = 0
            test_result_list = glob(save_dir1 + '/*')
            test_result_list = sorted(test_result_list, key=lambda x: int(x.split('/')[-1]))
            if len(test_result_list) > 5:
                for i in range(len(test_result_list) - 5):
                    shutil.rmtree(test_result_list[i])
                    print('remove {0} !'.format(test_result_list[i]))
            test_result = os.path.join(save_dir1, str(epoch + 1))
            if not os.path.exists(test_result):
                os.mkdir(test_result)
            for n, (input_image, target) in test_ds.enumerate():
                prediction = model(input_image, training=False)
                prediction = tf.clip_by_value(prediction, 0, 1)
                test_psnr += tf.reduce_mean(tf.image.psnr(target, prediction, max_val=1.0))
                # if n <= 10:
                prediction = np.squeeze(prediction) * 255
                input_image = np.squeeze(input_image) * 255
                target = np.squeeze(target) * 255
                joint_im = np.concatenate((input_image, prediction, target), axis=1)
                joint_im = Image.fromarray(np.uint8(joint_im))
                joint_im.save("{0}/{1:06d}.jpg".format(test_result, n + 1))
            n = tf.cast(n, dtype='float32')
            test_psnr /= (n + 1)
            with summary_writer.as_default():
                tf.summary.scalar('psnr', test_psnr, step=epoch)
            manager.save(checkpoint_number=epoch + 1)
            if test_psnr >= best_psnr[0]:
                checkpoint_list = glob(checkpoint_prefix + '-{}*'.format(epoch + 1))
                for file in checkpoint_list:
                    shutil.copy(file, '/'.join([checkpoint_best, file.split('/')[-1]]))
                shutil.copytree(test_result, save_dir2 + '/{:d}_{:2.2f}'.format(epoch + 1, best_psnr))
                best_result_list = glob(save_dir2 + '/*')
                best_result_list = sorted(best_result_list, key=lambda x: float(x.split('/')[-1].split('_')[1]))
                best_epoch = [int(epoch.split('/')[-1].split('_')[0]) for epoch in best_result_list]
                best_psnr = [float(epoch.split('/')[-1].split('_')[1]) for epoch in best_result_list]
                best_checkpoint_list = glob(checkpoint_best + '/*')
                if len(best_result_list) > 5:
                    for i in range(len(best_result_list) - 5):
                        shutil.rmtree(best_result_list[i])
                        remove_checkpoint = [remove_file for remove_file in best_checkpoint_list
                                             if int(remove_file.split('-')[1].split('.')[0]) == best_epoch[i]]
                        for j in range(3):
                            os.remove(remove_checkpoint[j])
                            print('remove {} !'.format(remove_checkpoint[j]))
                        print('remove {0} !'.format(best_result_list[i]))
            print('current psnr: {:2.2f}, best psnr: {:2.2f}, best epoch: {}'.format(test_psnr, best_psnr[-1], best_epoch[-1]))
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))


fit(train_dataset, 3000, test_dataset)

