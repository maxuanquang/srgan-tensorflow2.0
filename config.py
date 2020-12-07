from easydict import EasyDict as edict
import json
import os

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 8 # [16] use 8 if your GPU memory is small, and use [2, 4] in tl.vis.save_images / use 16 for faster training
config.TRAIN.lr_init = 1e-4
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 50
    # config.TRAIN.lr_decay_init = 0.1
    # config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)
config.TRAIN.number_of_images = 800
config.TRAIN.input_G_shape = (96, 96, 3)
config.TRAIN.input_D_shape = (384, 384, 3)
config.TRAIN.shuffle_buffer_size = 128
config.TRAIN.verbose = 10
config.TRAIN.n_epochs_save_model = 2

config.BASE_PATH = '/content/drive/MyDrive/ProjectDL/srgan' # chỉ cần thay đổi dòng này
config.SAVE_DIR = os.path.join(config.BASE_PATH, 'samples')
config.CHECKPOINT_DIR = os.path.join(config.BASE_PATH, 'models')
config.TRAIN.g_trained_dir = os.path.join(config.CHECKPOINT_DIR, 'g.h5')
config.TRAIN.d_trained_dir = os.path.join(config.CHECKPOINT_DIR, 'd.h5')
config.TRAIN.g_warmed_up_dir = os.path.join(config.CHECKPOINT_DIR, 'g_warmed_up.h5')

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 200
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = os.path.join(config.BASE_PATH, 'DIV2K/DIV2K_train_HR')
config.TRAIN.lr_img_path = os.path.join(config.BASE_PATH, 'DIV2K/DIV2K_train_LR_bicubic/X4/')

config.VALID = edict()
## test set location
config.VALID.hr_img_path = os.path.join(config.BASE_PATH, 'DIV2K/DIV2K_valid_HR')
config.VALID.lr_img_path = os.path.join(config.BASE_PATH, 'DIV2K/DIV2K_valid_LR_bicubic/X4')

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
