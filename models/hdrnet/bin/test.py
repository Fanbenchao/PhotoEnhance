import argparse
import logging
import cv2
import numpy as np
import os
import re
import setproctitle
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
tf.compat.v1.disable_eager_execution()
import sys
sys.path.append('/home/lupin/PhotoEnhance/hdrnet_lupin/hdrnet')
import models as models

logging.basicConfig(format="[%(process)d] %(levelname)s %(filename)s:%(lineno)s | %(message)s")
log = logging.getLogger("train")
log.setLevel(logging.INFO)

def get_dataset(path):
    dirname = os.path.dirname(path)
    with open(path, 'r') as fid:
        flist = [l.strip() for l in fid.readlines()]
    input_files = []
    output_files = []
    for f in flist:
        pre_name,img_name = f.split('_')[0], f.split('_')[1]
        input_files.append(os.path.join(dirname,'image',pre_name,img_name))
        output_files.append(os.path.join(dirname,'label',pre_name+'.JPG'))
    return input_files, output_files

def main(args,model_params):
    setproctitle.setproctitle('hdrnet_run')
    img_list,label_list = get_dataset(args.input)
    net_shape = model_params['net_input_size']
    t_fullres_input = tf.compat.v1.placeholder(tf.float32, (1, None, None, 3))
    t_lowres_input = tf.compat.v1.placeholder(tf.float32, (1, net_shape, net_shape, 3))
    mdl = getattr(models, model_params['model_name'])
    with tf.compat.v1.variable_scope('inference'):
        eval_prediction = mdl.inference(
            t_lowres_input, t_fullres_input,
            model_params, is_training=False)
    output = tf.cast(255.0*tf.squeeze(tf.clip_by_value(eval_prediction, 0, 1)), tf.uint8)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # Do not canibalize the entire GPU
    init = tf.compat.v1.global_variables_initializer()
    sv = tf.compat.v1.train.Supervisor(logdir=args.checkpoint_dir, init_op=init)
    with sv.managed_session() as sess:
        for i in tqdm(range(len(img_list))):
            im_input = cv2.imread(img_list[i], -1)  # -1 means read as is, no conversions.
            im_output = cv2.imread(label_list[i], -1)
            if im_input.shape[2] == 4:
                log.info("Input {} has 4 channels, dropping alpha".format(input_path))
                im_input = im_input[:, :, :3]
                im_output = im_output[:, :, :3]

            im_input = np.flip(im_input, 2)  # OpenCV reads BGR, convert back to RGB.
            im_output = np.flip(im_output, 2)

            im_input = skimage.img_as_float(im_input)
            im_output = skimage.img_as_float(im_output)

          # Make or Load lowres image
            if args.lowres_input is None:
                lowres_input = skimage.transform.resize(im_input, [net_shape, net_shape], order = 0)
            else:
                raise NotImplemented

            im_input = im_input[np.newaxis, :, :, :]
            lowres_input = lowres_input[np.newaxis, :, :, :]
            feed_dict = {t_fullres_input: im_input,t_lowres_input: lowres_input}

            out_ =  sess.run(output, feed_dict=feed_dict)

parser = argparse.ArgumentParser()
req_grp = parser.add_argument_group('required')
req_grp.add_argument('--checkpoint_dir', default= '/home/lupin/PhotoEnhance/hdrnet_lupin/hdrnet/checkpoint',type=str)
req_grp.add_argument('--input', default='/home/lupin/PhotoEnhance/dataset/SICE/Part2/test.txt',type=str)
req_grp.add_argument('--output', default=None)
req_grp.add_argument('--lowres_input', default=None, help='path to the lowres, TF inputs')
model_grp = parser.add_argument_group('model_params')
model_grp.add_argument('--model_name', default=models.__all__[0], type=str, help='classname of the model to use.', choices=models.__all__)
model_grp.add_argument('--net_input_size', default=256, type=int, help="size of the network's lowres image input.")
model_grp.add_argument('--batch_norm', dest='batch_norm', action='store_true', help='normalize batches. If False, uses the moving averages.')
model_grp.add_argument('--nobatch_norm', dest='batch_norm', action='store_false')
model_grp.add_argument('--channel_multiplier', default=1, type=int,  help='Factor to control net throughput (number of intermediate channels).')
model_grp.add_argument('--guide_complexity', default=16, type=int,  help='Control complexity of the guide network.')
model_grp.add_argument('--luma_bins', default=8, type=int,  help='Number of BGU bins for the luminance.')
model_grp.add_argument('--spatial_bin', default=16, type=int,  help='Size of the spatial BGU bins (pixels).')
parser.set_defaults(batch_norm = True)
args = parser.parse_args(args = [])
model_params = {}
for a in model_grp._group_actions:
    model_params[a.dest] = getattr(args, a.dest, None)
main(args,model_params)