import os
import time
import random
import numpy as np
import scipy, multiprocessing
from tensorflow.keras.models import model_from_json
from tensorflow import keras
import tensorflow as tf
import tensorlayer as tl
from model import get_G, get_D
from config import config
import cv2
from imutils.paths import list_images
import math

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size  # use 8 if your GPU memory is small, and change [4, 4] in tl.vis.save_images to [2, 4]
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1

## initialize G - khoi dong G
n_epoch_init = config.TRAIN.n_epoch_init
number_of_images = config.TRAIN.number_of_images
input_G_shape = config.TRAIN.input_G_shape
input_D_shape = config.TRAIN.input_D_shape
verbose = config.TRAIN.verbose
n_epochs_save_model = config.TRAIN.n_epochs_save_model
g_trained_dir = config.TRAIN.g_trained_dir
d_trained_dir = config.TRAIN.d_trained_dir
g_warmed_up_dir = config.TRAIN.g_warmed_up_dir
g_evaluate_dir = config.TRAIN.g_evaluate_dir
g_losses_txt = config.TRAIN.g_losses_txt
d_losses_txt = config.TRAIN.d_losses_txt

adv_coef = config.TRAIN.adv_coef
vgg_coef = config.TRAIN.vgg_coef
loss_type = config.TRAIN.loss_type

## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
shuffle_buffer_size = config.TRAIN.shuffle_buffer_size

# ni = int(np.sqrt(batch_size))

# create folders to save result images and trained models
save_dir = config.SAVE_DIR
tl.files.exists_or_mkdir(save_dir)
checkpoint_dir = config.CHECKPOINT_DIR
tl.files.exists_or_mkdir(checkpoint_dir)

def get_train_data():    
    train_hr_img_list = tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False)[:number_of_images]
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # (1356, 2040, 3)
    def generator_train():
        for img in train_hr_imgs:
            yield img
    
    def _map_fn_train(img):
        hr_patch = tf.image.random_crop(img, input_D_shape)
        # chuyen phan bo ve -1 +1
        hr_patch = hr_patch/(255./2.)
        hr_patch = hr_patch - 1.
        hr_patch = tf.image.random_flip_left_right(hr_patch)
        lr_patch = tf.image.resize(hr_patch, size=input_G_shape[:2])
        return lr_patch, hr_patch

    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=2) # prefetch truoc 2 batch
    return train_ds

def warmup():
    '''
    Warm-up G (Generator) using Mean Absolute Error Loss
    '''
    G = get_G(input_G_shape)

    lr_v = tf.Variable(lr_init)
    g_optimizer_init=tf.optimizers.Adam(lr_v, beta_1=beta1)

    train_ds = get_train_data()
    total_images = len(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False)[:number_of_images])
    n_step_epoch = math.ceil(total_images / batch_size)

    # initialize learning
    for epoch in range(n_epoch_init):
        for step,(lr_patchs,hr_patchs) in enumerate(train_ds):
            step_time = time.time()
            with tf.GradientTape() as tape:
                fake_hr_patchs = G(lr_patchs)
                mae_loss = tf.keras.losses.mean_absolute_error(fake_hr_patchs, hr_patchs)
            grad = tape.gradient(mae_loss, G.trainable_weights)
            g_optimizer_init.apply_gradients(zip(grad,G.trainable_weights))
            if (step == 0) or ((step+1) % verbose == 0):
                print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mae: {:.3f} ".format(
                    epoch+1, n_epoch_init, step+1, n_step_epoch, time.time() - step_time, np.mean(mae_loss)))
        if (epoch!=0) and ((epoch+1)%n_epochs_save_model==0):
            G.save_weights(os.path.join(checkpoint_dir, 'g_warmed_up_{}.h5'.format(epoch+1)))
    G.save_weights(os.path.join(checkpoint_dir, 'g_warmed_up.h5'))

def train():
    '''
    There are 5 loss types:
    'MAE', 'MSE', 'VGG', 'MAEVGG', 'MSEVGG'
    '''
    G = get_G(input_G_shape)
    G.load_weights(g_warmed_up_dir)
    D = get_D(input_D_shape)

    VGG = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=max,
        classes=2,
    )
    # Save architecture and weight
    with open(os.path.join(checkpoint_dir, 'VGG.json'), 'w') as f:
        f.write(VGG.to_json())
    VGG.save_weights(os.path.join(checkpoint_dir, 'VGG.h5'))

    lr_v = tf.Variable(lr_init)
    g_optimizer=tf.optimizers.Adam(lr_v, beta_1=beta1)
    d_optimizer=tf.optimizers.Adam(lr_v, beta_1=beta1)

    train_ds = get_train_data()
    total_images = len(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False)[:number_of_images])
    n_step_epoch = math.ceil(total_images / batch_size)

    # adversarial learning (G,D)
    for epoch in range(n_epoch):
        g_losses = []
        d_losses = []
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            step_time = time.time()
            # To compute multiple gradients over the same computation, create a persistent gradient tape
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = G(lr_patchs)
                logits_fake = D(fake_patchs)
                logits_real = D(hr_patchs)
                feature_fake = VGG((fake_patchs+1)/2.) # the pre-trained VGG uses the input range of [0, 1] but we use input range of [-1 1]
                feature_real = VGG((hr_patchs+1)/2.)
                
                d_loss1 = tf.keras.losses.binary_crossentropy(tf.ones_like(logits_real),logits_real, from_logits=True)
                d_loss2 = tf.keras.losses.binary_crossentropy(tf.zeros_like(logits_fake),logits_fake,from_logits=True)
                d_loss1 = tf.reduce_mean(d_loss1)
                d_loss2 = tf.reduce_mean(d_loss2)
                d_loss = d_loss1 + d_loss2
                g_gan_loss = tf.multiply(tf.constant(adv_coef),tf.keras.losses.binary_crossentropy(tf.ones_like(logits_fake),logits_fake,from_logits=True))
                g_gan_loss = tf.reduce_mean(g_gan_loss)
                
                if loss_type=='MSEVGG':
                    mse_loss = tf.keras.losses.mean_squared_error(fake_patchs, hr_patchs)
                    mse_loss = tf.reduce_mean(mse_loss)
                    vgg_loss = tf.multiply(tf.constant(vgg_coef),tf.keras.losses.mean_squared_error(feature_fake, feature_real))
                    vgg_loss = tf.reduce_mean(vgg_loss)
                    g_loss = 0.5*mse_loss + 0.5*vgg_loss + g_gan_loss
                    if (step == 0) or ((step+1) % verbose == 0):
                        print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss:{:.3f} (mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss:{:.3f} (d_loss1:{:.3f} d_loss2:{:.3f})".format(
                                epoch+1, n_epoch, step+1, n_step_epoch, time.time() - step_time, g_loss, mse_loss, vgg_loss, g_gan_loss, d_loss, d_loss1, d_loss2))
                elif loss_type=='MAEVGG':
                    mae_loss = tf.keras.losses.mean_absolute_error(fake_patchs, hr_patchs)
                    mae_loss = tf.reduce_mean(mae_loss)
                    vgg_loss = tf.multiply(tf.constant(vgg_coef),tf.keras.losses.mean_squared_error(feature_fake, feature_real))
                    vgg_loss = tf.reduce_mean(vgg_loss)
                    g_loss = 0.5*mae_loss + 0.5*vgg_loss + g_gan_loss
                    if (step == 0) or ((step+1) % verbose == 0):
                        print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss:{:.3f} (mae:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss:{:.3f} (d_loss1:{:.3f} d_loss2:{:.3f})".format(
                                epoch+1, n_epoch, step+1, n_step_epoch, time.time() - step_time, g_loss, mae_loss, vgg_loss, g_gan_loss, d_loss, d_loss1, d_loss2))
                elif loss_type=='MAE':
                    mae_loss = tf.keras.losses.mean_absolute_error(fake_patchs, hr_patchs)
                    mae_loss = tf.reduce_mean(mae_loss)
                    g_loss = mae_loss + g_gan_loss
                    if (step == 0) or ((step+1) % verbose == 0):
                        print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss:{:.3f} (mae:{:.3f}, adv:{:.3f}) d_loss: {:.3f} (d_loss1: {:.3f} d_loss2: {:.3f})".format(
                                epoch+1, n_epoch, step+1, n_step_epoch, time.time() - step_time, g_loss, mae_loss, g_gan_loss, d_loss, d_loss1, d_loss2))
                elif loss_type=='VGG':
                    vgg_loss = tf.keras.losses.mean_squared_error(feature_fake, feature_real)
                    vgg_loss = tf.reduce_mean(vgg_loss)
                    g_loss = vgg_loss + g_gan_loss
                    if (step == 0) or ((step+1) % verbose == 0):
                        print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss:{:.3f} (vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f} (d_loss1: {:.3f} d_loss2: {:.3f})".format(
                                epoch+1, n_epoch, step+1, n_step_epoch, time.time() - step_time, g_loss, vgg_loss, g_gan_loss, d_loss, d_loss1, d_loss2))
                elif loss_type=='MSE':
                    mse_loss = tf.keras.losses.mean_squared_error(fake_patchs, hr_patchs)
                    mse_loss = tf.reduce_mean(mse_loss)
                    g_loss = mse_loss + g_gan_loss
                    if (step == 0) or ((step+1) % verbose == 0):
                        print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss:{:.3f} (mse:{:.3f}, adv:{:.3f}) d_loss: {:.3f} (d_loss1: {:.3f} d_loss2: {:.3f})".format(
                                epoch+1, n_epoch, step+1, n_step_epoch, time.time() - step_time, g_loss, mse_loss, g_gan_loss, d_loss, d_loss1, d_loss2))
            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))

            g_losses.append(g_loss)
            d_losses.append(d_loss)

        # save losses to file
        with open(g_losses_txt, 'a') as f:
            for item in g_losses:
                f.write("{}\n".format(item))
                
        with open(d_losses_txt, 'a') as f:
            for item in d_losses:
                f.write("{}\n".format(item))
        
        # update the learning rate
        if (epoch != 0) and ((epoch+1) % decay_every == 0):
            new_lr_decay = lr_decay**((epoch+1) // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        if (epoch != 0) and ((epoch+1) % n_epochs_save_model == 0):
            tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_{}.png'.format(epoch+1)))
            G.save_weights(os.path.join(checkpoint_dir, 'g_{}.h5'.format(epoch+1)))
            D.save_weights(os.path.join(checkpoint_dir, 'd_{}.h5'.format(epoch+1)))

def evaluate():
    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## if your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)

    ###========================== DEFINE MODEL ============================###
    imid = 1  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]

    G = get_G([None, None, 3])
    G.load_weights(g_evaluate_dir)

    valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
    valid_lr_img = valid_lr_img[np.newaxis,:,:,:]
    size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]

    out = G(valid_lr_img).numpy()

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], os.path.join(save_dir, 'valid_gen.png'))
    tl.vis.save_image(valid_lr_img[0], os.path.join(save_dir, 'valid_lr.png'))
    tl.vis.save_image(valid_hr_img, os.path.join(save_dir, 'valid_hr.png'))

    out_bicu = cv2.resize(valid_lr_img[0], (size[1] * 4, size[0] * 4))
    tl.vis.save_image(out_bicu, os.path.join(save_dir, 'valid_bicubic.png'))

def train_continue():
    G = get_G(input_G_shape)
    D = get_D(input_D_shape)
    G.load_weights(g_trained_dir)
    D.load_weights(d_trained_dir)

    # Load trained model
    json_file = open(os.path.join(checkpoint_dir, 'VGG.json'), 'r')
    VGG_json = json_file.read()
    json_file.close()
    VGG = model_from_json(VGG_json)
    VGG.load_weights(os.path.join(checkpoint_dir, 'VGG.h5'))

    lr_v = tf.Variable(lr_init)
    g_optimizer=tf.optimizers.Adam(lr_v, beta_1=beta1)
    d_optimizer=tf.optimizers.Adam(lr_v, beta_1=beta1)

    train_ds = get_train_data()
    total_images = len(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False)[:number_of_images])
    n_step_epoch = math.ceil(total_images / batch_size)
    
    # adversarial learning (G,D)
    for epoch in range(n_epoch):
        g_losses = []
        d_losses = []
        for step, (lr_patchs, hr_patchs) in enumerate(train_ds):
            step_time = time.time()
            # To compute multiple gradients over the same computation, create a persistent gradient tape
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = G(lr_patchs)
                logits_fake = D(fake_patchs)
                logits_real = D(hr_patchs)
                feature_fake = VGG((fake_patchs+1)/2.) # the pre-trained VGG uses the input range of [0, 1] but we use input range of [-1 1]
                feature_real = VGG((hr_patchs+1)/2.)
                
                d_loss1 = tf.keras.losses.binary_crossentropy(tf.ones_like(logits_real),logits_real, from_logits=True)
                d_loss2 = tf.keras.losses.binary_crossentropy(tf.zeros_like(logits_fake),logits_fake,from_logits=True)
                d_loss1 = tf.reduce_mean(d_loss1)
                d_loss2 = tf.reduce_mean(d_loss2)
                d_loss = d_loss1 + d_loss2
                g_gan_loss = tf.multiply(tf.constant(adv_coef),tf.keras.losses.binary_crossentropy(tf.ones_like(logits_fake),logits_fake,from_logits=True))
                g_gan_loss = tf.reduce_mean(g_gan_loss)
                
                if loss_type=='MSEVGG':
                    mse_loss = tf.keras.losses.mean_squared_error(fake_patchs, hr_patchs)
                    mse_loss = tf.reduce_mean(mse_loss)
                    vgg_loss = tf.multiply(tf.constant(vgg_coef),tf.keras.losses.mean_squared_error(feature_fake, feature_real))
                    vgg_loss = tf.reduce_mean(vgg_loss)
                    g_loss = 0.5*mse_loss + 0.5*vgg_loss + g_gan_loss
                    if (step == 0) or ((step+1) % verbose == 0):
                        print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss:{:.3f} (mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss:{:.3f} (d_loss1:{:.3f} d_loss2:{:.3f})".format(
                                epoch+1, n_epoch, step+1, n_step_epoch, time.time() - step_time, g_loss, mse_loss, vgg_loss, g_gan_loss, d_loss, d_loss1, d_loss2))
                elif loss_type=='MAEVGG':
                    mae_loss = tf.keras.losses.mean_absolute_error(fake_patchs, hr_patchs)
                    mae_loss = tf.reduce_mean(mae_loss)
                    vgg_loss = tf.multiply(tf.constant(vgg_coef),tf.keras.losses.mean_squared_error(feature_fake, feature_real))
                    vgg_loss = tf.reduce_mean(vgg_loss)
                    g_loss = 0.5*mae_loss + 0.5*vgg_loss + g_gan_loss
                    if (step == 0) or ((step+1) % verbose == 0):
                        print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss:{:.3f} (mae:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss:{:.3f} (d_loss1:{:.3f} d_loss2:{:.3f})".format(
                                epoch+1, n_epoch, step+1, n_step_epoch, time.time() - step_time, g_loss, mae_loss, vgg_loss, g_gan_loss, d_loss, d_loss1, d_loss2))
                elif loss_type=='MAE':
                    mae_loss = tf.keras.losses.mean_absolute_error(fake_patchs, hr_patchs)
                    mae_loss = tf.reduce_mean(mae_loss)
                    g_loss = mae_loss + g_gan_loss
                    if (step == 0) or ((step+1) % verbose == 0):
                        print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss:{:.3f} (mae:{:.3f}, adv:{:.3f}) d_loss: {:.3f} (d_loss1: {:.3f} d_loss2: {:.3f})".format(
                                epoch+1, n_epoch, step+1, n_step_epoch, time.time() - step_time, g_loss, mae_loss, g_gan_loss, d_loss, d_loss1, d_loss2))
                elif loss_type=='VGG':
                    vgg_loss = tf.keras.losses.mean_squared_error(feature_fake, feature_real)
                    vgg_loss = tf.reduce_mean(vgg_loss)
                    g_loss = vgg_loss + g_gan_loss
                    if (step == 0) or ((step+1) % verbose == 0):
                        print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss:{:.3f} (vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f} (d_loss1: {:.3f} d_loss2: {:.3f})".format(
                                epoch+1, n_epoch, step+1, n_step_epoch, time.time() - step_time, g_loss, vgg_loss, g_gan_loss, d_loss, d_loss1, d_loss2))
                elif loss_type=='MSE':
                    mse_loss = tf.keras.losses.mean_squared_error(fake_patchs, hr_patchs)
                    mse_loss = tf.reduce_mean(mse_loss)
                    g_loss = mse_loss + g_gan_loss
                    if (step == 0) or ((step+1) % verbose == 0):
                        print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss:{:.3f} (mse:{:.3f}, adv:{:.3f}) d_loss: {:.3f} (d_loss1: {:.3f} d_loss2: {:.3f})".format(
                                epoch+1, n_epoch, step+1, n_step_epoch, time.time() - step_time, g_loss, mse_loss, g_gan_loss, d_loss, d_loss1, d_loss2))
            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))

            g_losses.append(g_loss)
            d_losses.append(d_loss)
        
        # save losses to file
        with open(g_losses_txt, 'a') as f:
            for item in g_losses:
                f.write("{}\n".format(item))
                
        with open(d_losses_txt, 'a') as f:
            for item in d_losses:
                f.write("{}\n".format(item))
            
        # update the learning rate
        if (epoch != 0) and ((epoch+1) % decay_every == 0):
            new_lr_decay = lr_decay**((epoch+1) // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)

        if (epoch != 0) and ((epoch+1) % n_epochs_save_model == 0):
            tl.vis.save_images(fake_patchs.numpy(), [2, 4], os.path.join(save_dir, 'train_g_{}.png'.format(epoch+1)))
            G.save_weights(os.path.join(checkpoint_dir, 'g_{}.h5'.format(epoch+1)))
            D.save_weights(os.path.join(checkpoint_dir, 'd_{}.h5'.format(epoch+1)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'warmup':
        warmup()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    elif tl.global_flag['mode'] == 'continue':
        train_continue()
    else:
        raise Exception("Unknow --mode")
