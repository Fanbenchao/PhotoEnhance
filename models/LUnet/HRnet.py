import tensorflow as tf
import numpy as np
import logging
logger = logging.getLogger(__name__)

def Bottleneck(x,filter,expansion = 4,stride = 1,downsample = None):
    residual = x
    out = tf.keras.layers.Conv2D(filter, 1, padding='same', use_bias=False)(x)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv2D(filter, 3, strides=stride, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv2D(filter * expansion, 1, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    if downsample is not None:
        residual = downsample

    out = out + residual
    out = tf.keras.layers.ReLU()(out)
    return out

def BasicBlock(x,filter,expansion = 1,stride = 1,downsample = None):
    residual = x
    out = tf.keras.layers.Conv2D(filter, 3,strides=stride, padding='same', use_bias=False)(x)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Conv2D(filter, 3, padding='same', use_bias=False)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    if downsample is not None:
        residual = downsample
    out = out + residual
    out = tf.keras.layers.ReLU()(out)
    return out

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}
expansion_dict = {
    'BASIC': 1,
    'BOTTLENECK': 4
}

def _make_layer(x, block_name, inplanes, planes, blocks, stride=1):
    downsample = None
    expansion = expansion_dict[block_name]
    if stride != 1 or inplanes != planes * expansion:
        downsample = tf.keras.layers.Conv2D(planes*expansion, 1, strides= stride, padding='same', use_bias=False)(x)
        downsample = tf.keras.layers.BatchNormalization()(downsample)

    block = blocks_dict[block_name]
    x = block(x,planes,stride = stride,downsample = downsample)
    for i in range(1,blocks):
        x = block(x,planes)
    return [x]
def _make_transition_layer(x,num_channels_pre_layer, num_channels_cur_layer):
    num_branches_cur = len(num_channels_cur_layer)
    num_branches_pre = len(num_channels_pre_layer)
    transition_layers = []
    for i in range(num_branches_cur):
        if i < num_branches_pre:
            if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                out = tf.keras.layers.Conv2D(num_channels_cur_layer[i], 3, strides= 1, padding='same', use_bias=False)(x[i])
                out = tf.keras.layers.BatchNormalization()(out)
                out = tf.keras.layers.ReLU()(out)
                transition_layers.append(out)
            else:
                transition_layers.append(x[i])
        else:
            for j in range(i+1-num_branches_pre):
                outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else num_channels_pre_layer[-1]
                out = tf.keras.layers.Conv2D(outchannels, 3, strides= 2, padding='same', use_bias=False)(x[-1])
                out = tf.keras.layers.BatchNormalization()(out)
                out = tf.keras.layers.ReLU()(out)
                transition_layers.append(out)
    return transition_layers

def _make_one_branch(x, branch_index, block_name, num_blocks, num_inchannels, num_channels, stride=1):
    down = None
    expansion = expansion_dict[block_name]
    block = blocks_dict[block_name]
    if stride != 1 or num_inchannels[branch_index] != num_channels[branch_index]*expansion:
        down = tf.keras.layers.Conv2D(num_channels[branch_index]*expansion, 1,strides=stride, padding='same', use_bias=False)(x)
        down = tf.keras.layers.BatchNormalization()(down)
    out = block(x,num_channels[branch_index],stride = stride, downsample= down)
    for i in range(1, num_blocks[branch_index]):
        out = block(out,num_channels[branch_index])
    return out


def _make_branches(x, num_branches, block, num_blocks, num_inchannels, num_channels):
    branches = []
    for i in range(num_branches):
        branches.append(_make_one_branch(x[i],i, block, num_blocks, num_inchannels, num_channels))

    return branches

def _check_branches(num_branches, num_blocks,num_inchannels, num_channels):
    if num_branches != len(num_blocks):
        error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
            num_branches, len(num_blocks))
        logger.error(error_msg)
        raise ValueError(error_msg)

    if num_branches != len(num_channels):
        error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
            num_branches, len(num_channels))
        logger.error(error_msg)
        raise ValueError(error_msg)

    if num_branches != len(num_inchannels):
        error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
            num_branches, len(num_inchannels))
        logger.error(error_msg)
        raise ValueError(error_msg)

def _make_fuse_layers(x, num_branches, num_inchannels,multi_scale_output):
    if num_branches == 1:
        return None
    fuse_layers = []
    for i in range(num_branches if multi_scale_output else 1):
        fuse_layer = []
        for j in range(num_branches):
            out1 = x[j]
            out2 = x[j]
            if j > i:
                out1 = tf.keras.layers.Conv2D(num_inchannels[i], 1, padding='same', use_bias=False)(out1)
                out1 = tf.keras.layers.BatchNormalization()(out1)
                fuse_layer.append(out1)
            elif j == i:
                fuse_layer.append(None)
            else:
                for k in range(i-j):
                    if k == i - j - 1:
                        out2 = tf.keras.layers.Conv2D(num_inchannels[i], 3, 2, padding='same', use_bias=False)(out2)
                        out2 = tf.keras.layers.BatchNormalization()(out2)
                    else:
                        out2 = tf.keras.layers.Conv2D(num_inchannels[j], 3, 2, padding='same', use_bias=False)(out2)
                        out2 = tf.keras.layers.BatchNormalization()(out2)
                        out2 = tf.keras.layers.ReLU()(out2)
                fuse_layer.append(out2)
        fuse_layers.append(fuse_layer)

    return fuse_layers

def _make_stage(x,layer_config,num_inchannels,multi_scale_output = True):
    out = x
    num_modules = layer_config['num_modules']
    num_branches = layer_config['num_branches']
    num_blocks = layer_config['num_blocks']
    num_channels = layer_config['num_channels']
    block_name = layer_config['block']
    fuse_method = layer_config['fuse_method']
    _check_branches(num_branches, num_blocks, num_inchannels, num_channels)

    for num in range(num_modules):
        if not multi_scale_output and num == num_modules - 1:
            reset_multi_scale_output = False
        else:
            reset_multi_scale_output = True

        out = _make_branches(out, num_branches, block_name, num_blocks, num_inchannels, num_channels)
        x_fuse = []
        fuse_layers = _make_fuse_layers(out, num_branches, num_inchannels,reset_multi_scale_output)
        for i in range(len(fuse_layers)):
            y = out[0] if i == 0 else fuse_layers[i][0]
            for j in range(1,num_branches):
                if i == j:
                    y = y + out[j]
                elif j > i:
                    y = y + tf.image.resize(fuse_layers[i][j], size = [out[i].shape[1],out[i].shape[2]])
                    # height_output = out[i].shape[1]//out[j].shape[1]
                    # width_output = out[i].shape[2]//out[j].shape[2]
                    # y = y + tf.keras.layers.UpSampling2D(size=(height_output, width_output), interpolation='bilinear')(fuse_layers[i][j])
                else:
                    y = y + fuse_layers[i][j]
            y = tf.keras.layers.ReLU()(y)
            x_fuse.append(y)
        out = x_fuse
    return out, num_inchannels


def downsample(x,filters, size, strides = 2, apply_batchnorm=True):
    out = tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same', use_bias=False)(x)
    if apply_batchnorm:
        out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    return out

def ResidualBlock(x,channels):
    residual = x
    x = tf.keras.layers.Conv2D(channels, 3, padding='same')(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(channels, 3, padding='same')(x)
    return residual+x
def RsidualGroup(x,channels,num_blocks):
    for i in range(num_blocks):
        x = ResidualBlock(x,channels)
    return x

def DDF_Block(x,feature_mem_up,down_convs,up_convs):
    ft_fusion = x
    num_ft = len(feature_mem_up)
    for i in range(num_ft):
        ft = ft_fusion
        tensor_shape = []
        for j in range(num_ft-i):
            tensor_shape.append(ft.shape)
            ft = down_convs[num_ft-1-j](ft)
        ft = ft - feature_mem_up[i]
        for j in range(num_ft-i):
            ft = up_convs[j+i](ft)
            ft = tf.image.resize(ft, size=[tensor_shape[-j-1][1], tensor_shape[-j-1][2]])
        ft_fusion = ft_fusion+ft
    return ft_fusion

def HRnet():
    stage1_config = {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': [4],
                     'num_channels': [64], 'fuse_method': 'sum'}
    stage2_config = {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': [4, 4],
                     'num_channels': [48, 96], 'fuse_method': 'sum'}
    stage3_config = {'num_modules': 3, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': [4, 4, 4],
                     'num_channels': [48, 96,192], 'fuse_method': 'sum'}
    stage4_config = {'num_modules': 2, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': [4, 4, 4, 4],
                     'num_channels': [48, 96, 192, 384], 'fuse_method': 'sum'}
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    down = downsample(inputs,64,3,strides=1)
    down = downsample(down,64,3,strides=1)

    #stage1
    num_channels = stage1_config['num_channels'][0]
    block_name = stage1_config['block']
    num_blocks = stage1_config['num_blocks'][0]
    expansion = expansion_dict[block_name]

    layer1 = _make_layer(down,block_name, 64, num_channels, num_blocks)
    stage1_out_channel = expansion * num_channels

    #stage2
    num_channels = stage2_config['num_channels']
    block_name = stage2_config['block']
    expansion = expansion_dict[block_name]
    num_channels = [num_channels[i] * expansion for i in range(len(num_channels))]
    transition1 = _make_transition_layer(layer1, [stage1_out_channel], num_channels)
    stage2,pre_stage_channels = _make_stage(transition1, stage2_config, num_channels)

    #stage3
    num_channels = stage3_config['num_channels']
    block_name = stage3_config['block']
    expansion = expansion_dict[block_name]
    num_channels = [num_channels[i] * expansion for i in range(len(num_channels))]
    transition2 = _make_transition_layer(stage2, pre_stage_channels, num_channels)
    stage3, pre_stage_channels = _make_stage(transition2, stage3_config, num_channels)

    #stage4
    num_channels = stage4_config['num_channels']
    block_name = stage4_config['block']
    expansion = expansion_dict[block_name]
    num_channels = [num_channels[i] * expansion for i in range(len(num_channels))]
    transition3 = _make_transition_layer(stage3, pre_stage_channels, num_channels)
    stage4, pre_stage_channels = _make_stage(transition3, stage4_config, num_channels)

    # last_inp_channels = np.int(np.sum(pre_stage_channels))
    #layer split
    x0 = stage4[0]
    x1 = stage4[1]
    x2 = stage4[2]
    x3 = stage4[3]

    #define up_conv and down_conv layer
    down_convs = []
    up_convs = []
    channels = [x3.shape[-1], x2.shape[-1], x1.shape[-1], x0.shape[-1]]
    for i in range(1,4):
        down_convs.append(tf.keras.layers.Conv2D(channels[i-1], 4, strides=2, padding='same', activation = 'relu'))
        up_convs.append(tf.keras.layers.Conv2DTranspose(channels[i], 4, strides=2,padding='same', activation = 'relu'))
    #fusion1
    res16x = x3
    in_ft = res16x*2
    res16x = RsidualGroup(in_ft,channels[0],4)+in_ft-res16x
    feature_mem_up = [res16x]

    #fusion2
    res16x = tf.keras.layers.Conv2DTranspose(channels[1], 3,strides=2,padding='same')(res16x)
    res16x = tf.image.resize(res16x, size = [x2.shape[1],x2.shape[2]])
    res8x = res16x+x2
    res8x = RsidualGroup(res8x,channels[1],3)+res8x-res16x
    res8x = DDF_Block(res8x,feature_mem_up,down_convs,up_convs)
    feature_mem_up.append(res8x)

    #fusion3
    res8x = tf.keras.layers.Conv2DTranspose(channels[2], 3,strides=2,padding='same')(res8x)
    res8x = tf.image.resize(res8x, size=[x1.shape[1], x1.shape[2]])
    res4x = res8x + x1
    res4x = RsidualGroup(res4x,channels[2],3)+res4x-res8x
    res4x = DDF_Block(res4x,feature_mem_up,down_convs,up_convs)
    feature_mem_up.append(res4x)

    #fusion4
    res4x = tf.keras.layers.Conv2DTranspose(channels[3], 3,strides=2,padding='same')(res4x)
    res4x = tf.image.resize(res4x, size=[x0.shape[1], x0.shape[2]])
    res2x = res4x + x0
    res2x = RsidualGroup(res2x,channels[3],3)+res2x-res4x
    res2x = DDF_Block(res2x,feature_mem_up,down_convs,up_convs)

    #output layer
    output_layer = tf.keras.layers.Conv2D(3, 3, padding='same')(res2x)
    # [48, 96, 192, 384]

    # x0 = stage4[0]
    # x1 = tf.keras.layers.UpSampling2D(size=(x1_h, x1_w), interpolation='bilinear')(stage4[1])
    # x2 = tf.keras.layers.UpSampling2D(size=(x2_h, x2_w), interpolation='bilinear')(stage4[2])
    # x3 = tf.keras.layers.UpSampling2D(size=(x3_h, x3_w), interpolation='bilinear')(stage4[3])
    #
    # x = x = tf.keras.layers.Concatenate()([x0, x1, x2, x3])
    # last_layer = tf.keras.layers.Conv2D(last_inp_channels, 1, 1, padding='same')(x)
    # last_layer = tf.keras.layers.BatchNormalization()(last_layer)
    # last_layer = tf.keras.layers.ReLU()(last_layer)
    # last_layer = tf.keras.layers.Conv2D(3, 1, 1, padding='same')(x)

    return tf.keras.Model(inputs=inputs, outputs=output_layer)






