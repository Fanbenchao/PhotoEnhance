import os
import time
import tensorflow as tf
import datetime
import numpy as np
from PIL import Image
import shutil
from data_loader_people import load_image_train, load_image_test
from data_loader_low import file_based_input_fn_builder
from HRnet import HRnet
from model import Generator, Discriminator
from loss import generator_loss, discriminator_loss
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

beta = 100
buffer_size = 100
train_batch_size = 2
test_batch_size = 1
# train_path = '../dataset/SICE/Part1/train.tfrecords'
# test_path = '../dataset/SICE/Part1/valid.tfrecords'
# train_dataset = file_based_input_fn_builder(train_path,train_batch_size,is_training = True,buffer_size = buffer_size)
# test_dataset = file_based_input_fn_builder(test_path,test_batch_size,is_training = False)
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

train_img_list = tf.data.Dataset.list_files(train_img_files,shuffle= False)
train_label_list = tf.data.Dataset.list_files(train_label_files,shuffle= False)
train_dataset = tf.data.Dataset.zip((train_img_list,train_label_list))
train_dataset = train_dataset.map(load_image_train,num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size)
train_dataset = train_dataset.batch(train_batch_size)

test_img_list = tf.data.Dataset.list_files(test_img_files,shuffle= False)
test_label_list = tf.data.Dataset.list_files(test_label_files,shuffle= False)
test_dataset = tf.data.Dataset.zip((test_img_list,test_label_list))
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(test_batch_size)



generator = Generator()
# generator = HRnet()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)


checkpoint_dir = './test'
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint_best  = os.path.join(checkpoint_dir,'best')
if not os.path.exists(checkpoint_best):
    os.mkdir(checkpoint_best)
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)

save_dir1 = '/'.join([checkpoint_dir,'test_result'])
save_dir2 = '/'.join([checkpoint_dir,'best_result'])
if not os.path.exists(save_dir1):
    os.mkdir(save_dir1)
if not os.path.exists(save_dir2):
    os.mkdir(save_dir2)
log_dir = '/'.join([checkpoint_dir,'logs'])

summary_writer = tf.summary.create_file_writer(log_dir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
@tf.function
def train_step(input_image, target,LAMBDA):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target,LAMBDA)
    disc_loss,real_loss,generated_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  return gen_total_loss, gen_gan_loss, gen_l1_loss,disc_loss,real_loss,generated_loss
@tf.function
def gen_step(input_image, target,LAMBDA):
    with tf.GradientTape() as gen_tape:
        gen_output = generator(input_image, training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target,LAMBDA)
        
    generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
    return gen_total_loss,gen_gan_loss, gen_l1_loss


@tf.function
def dis_step(input_image, target):
    with tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        disc_loss,real_loss,generated_loss = discriminator_loss(disc_real_output, disc_generated_output)

    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))

    return disc_loss,real_loss,generated_loss
def fit(train_ds, end_epochs, test_ds):
    best_result_list = glob(save_dir2+'/*')
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
    for epoch in range(start_epochs,end_epochs):
        start = time.time()
        print("Epoch: ", epoch+1)
        
        psnr,gen_total_loss,gen_gan_loss, gen_l1_loss,disc_loss,real_loss,generated_loss = 0,0,0,0,0,0,0
        # Train
        for n, (input_image, target) in tqdm(train_ds.enumerate()):
            # gen_total_loss_, gen_gan_loss_, gen_l1_loss_,disc_loss_,real_loss_,generated_loss_ = train_step(input_image, target, LAMBDA)
            # _, _, _ = gen_step(input_image,target,LAMBDA)
            gen_total_loss_, gen_gan_loss_, gen_l1_loss_ = gen_step(input_image, target, beta)
            disc_loss_, real_loss_, generated_loss_ = dis_step(input_image, target)
            gen_total_loss_, gen_gan_loss_, gen_l1_loss_ = gen_step(input_image, target, beta)
            gen_total_loss += gen_total_loss_
            gen_gan_loss += gen_gan_loss_
            gen_l1_loss += gen_l1_loss_
            disc_loss += disc_loss_
            real_loss += real_loss_
            generated_loss += generated_loss_
        n = tf.cast(n,dtype= 'float32')
        gen_total_loss /= (n+1)
        gen_gan_loss /= (n+1)
        gen_l1_loss /= (n+1)
        disc_loss /= (n+1)
        real_loss /= (n+1)
        generated_loss /= (n+1)
        with summary_writer.as_default():
                tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
                tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
                tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
                tf.summary.scalar('disc_loss', disc_loss, step=epoch)
                tf.summary.scalar('real_loss', real_loss, step=epoch)
                tf.summary.scalar('generated_loss', generated_loss, step=epoch)
        print('gen_total_loss: {:4.4f}, disc_loss: {:4.4f}, gen_gan_loss: {:4.4f}, gen_l1_loss: {:4.4f},real_loss: {:4.4f},generated_loss: {:4.4f}'.format(
                gen_total_loss,disc_loss,gen_gan_loss,gen_l1_loss,real_loss,generated_loss))

        if (epoch + 1) % 50 == 0:
            test_psnr = 0
            test_result_list = glob(save_dir1+'/*')
            test_result_list = sorted(test_result_list, key=lambda x: int(x.split('/')[-1]))
            if len(test_result_list) > 5:
                for i in range(len(test_result_list)-5):
                    shutil.rmtree(test_result_list[i])
                    print('remove {0} !'.format(test_result_list[i]))
            test_result = os.path.join(save_dir1,str(epoch+1))
            if not os.path.exists(test_result):
                os.mkdir(test_result)
            for n, (input_image, target) in test_ds.enumerate():
                prediction = generator(input_image, training=False)
                prediction = tf.clip_by_value(prediction,0,1)
                test_psnr += tf.reduce_mean(tf.image.psnr(target,prediction,max_val = 1.0))
            # if n <= 10:
                prediction = np.squeeze(prediction)*255
                input_image = np.squeeze(input_image)*255
                target = np.squeeze(target)*255
                joint_im = np.concatenate((input_image, prediction, target), axis=1)
                joint_im = Image.fromarray(np.uint8(joint_im))
                joint_im.save("{0}/{1:06d}.jpg".format(test_result, n+1))
            n = tf.cast(n,dtype='float32')
            test_psnr /= (n+1)
            with summary_writer.as_default():
                tf.summary.scalar('psnr', test_psnr, step= epoch)
            manager.save(checkpoint_number = epoch+1)
            if test_psnr >= best_psnr[0]:
                checkpoint_list = glob(checkpoint_prefix+'-{}*'.format(epoch+1))
                for file in checkpoint_list:
                    shutil.copy(file,'/'.join([checkpoint_best,file.split('/')[-1]]))
                shutil.copytree(test_result, save_dir2 + '/{:d}_{:2.2f}'.format(epoch + 1, test_psnr))
                best_result_list = glob(save_dir2+'/*')
                best_result_list = sorted(best_result_list, key=lambda x: float(x.split('/')[-1].split('_')[1]))
                best_epoch = [int(epoch.split('/')[-1].split('_')[0]) for epoch in best_result_list]
                best_psnr = [float(epoch.split('/')[-1].split('_')[1]) for epoch in best_result_list]
                best_checkpoint_list = glob(checkpoint_best+'/*')
                if len(best_result_list) > 5:
                    for i in range(len(best_result_list) - 5):
                        shutil.rmtree(best_result_list[i])
                        remove_checkpoint = [remove_file for remove_file in best_checkpoint_list
                                             if int(remove_file.split('-')[1].split('.')[0]) == best_epoch[i]]
                        for j in range(3):
                            os.remove(remove_checkpoint[j])
                            print('remove {} !'.format(remove_checkpoint[j]))
                        print('remove {0} !'.format(best_result_list[i]))
            print('current psnr: {:2.2f}, best psnr: {:2.2f}, best epoch: {}'.format(test_psnr,best_psnr[-1],best_epoch[-1]))
        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,time.time()-start))
    # manager.save(checkpoint_number = epoch+1)
fit(train_dataset, 8000, test_dataset)

