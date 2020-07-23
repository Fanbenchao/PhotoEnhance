import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
def generate_images(model, test_input, tar,save_dir,img_num,epoch):
    save_dir = '/'.join([save_dir,str(epoch)])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    prediction = model(test_input, training = False)
    plt.figure(figsize=(15,15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.savefig(fname= '/'.join([save_dir,str(img_num)+'.jpg']))
#create multi image
img1_path = '../dataset/crop_people/image/1.jpg'
img2_path = '../dataset/crop_people/image/188.jpg'
img1 = imageio.imread(img1_path)
img2 = imageio.imread(img2_path)
img1 = img1[512:1024,512:1024,:]
img2 = img2[512:1024,720:1440,:]
img1_img = Image.fromarray(img1)
img2_img = Image.fromarray(img2)
tensor1 = tf.constant(img1/255.0)
tensor2 = tf.constant(img2/255.0)
#bright 0.1,-0.1
img1_bright = tf.image.adjust_brightness(tensor1,delta = -0.1)
img2_bright = tf.image.adjust_brightness(tensor2,delta = -0.1)
img1_bright = tf.clip_by_value(img1_bright,0,1)
img2_bright = tf.clip_by_value(img2_bright,0,1)
img1_bright = img1_bright*255
img2_bright = img2_bright*255
img1_bright = Image.fromarray(np.uint8(img1_bright))
img2_bright = Image.fromarray(np.uint8(img2_bright))
#contrast 2
img1_contrast = tf.image.adjust_contrast(tensor1,contrast_factor = 2)
img2_contrast = tf.image.adjust_contrast(tensor2,contrast_factor = 2)
img1_contrast = tf.clip_by_value(img1_contrast,0,1)
img2_contrast = tf.clip_by_value(img2_contrast,0,1)
img1_contrast = img1_contrast*255
img2_contrast = img2_contrast*255
img1_contrast = Image.fromarray(np.uint8(img1_contrast))
img2_contrast = Image.fromarray(np.uint8(img2_contrast))
#0.9,1.2
img1_saturation = tf.image.adjust_saturation(tensor1,saturation_factor = 0.9)
img2_saturation = tf.image.adjust_saturation(tensor2,saturation_factor = 0.9)
img1_saturation = tf.clip_by_value(img1_saturation,0,1)
img2_saturation = tf.clip_by_value(img2_saturation,0,1)
img1_saturation = img1_saturation*255
img2_saturation = img2_saturation*255
img1_saturation = Image.fromarray(np.uint8(img1_saturation))
img2_saturation = Image.fromarray(np.uint8(img2_saturation))
#1.8,0.7 
img1_gamma = tf.image.adjust_gamma(tensor1,gamma=0.6, gain=1)
img2_gamma = tf.image.adjust_gamma(tensor2,gamma=0.6, gain=1)
img1_gamma = tf.clip_by_value(img1_gamma,0,1)
img2_gamma = tf.clip_by_value(img2_gamma,0,1)
img1_gamma = img1_gamma*255
img2_gamma = img2_gamma*255
img1_gamma = Image.fromarray(np.uint8(img1_gamma))
img2_gamma = Image.fromarray(np.uint8(img2_gamma))