import tensorflow as tf
from functools import reduce
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d for d in tensor.get_shape()[1:]), 1)
def generator_loss(disc_generated_output, gen_output, target, LAMBDA):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss+generated_loss

    return total_disc_loss,real_loss,generated_loss
def L2_loss(target,prediction):
    return tf.reduce_mean(tf.square(target-prediction))
def compare_psnr(gen_output, target):
    psnr = tf.reduce_mean(tf.image.psnr(target,gen_output,max_val = 1.0))
    return psnr