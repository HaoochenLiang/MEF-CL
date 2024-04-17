import argparse
import os
import torch
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, default='./test_data', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=600, help='then crop to this size')
        self.parser.add_argument('--patchSize', type=int, default=64, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--dataset_mode', type=str, default='test', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument('--no_dropout', type=bool, default=True, help='no dropout for the generator')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='no', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--skip', type=float, default=1, help='B = net.forward(A) + skip*A')
        self.parser.add_argument('--use_mse', action='store_true', help='MSELoss')
        self.parser.add_argument('--l1', type=float, default=10.0, help='L1 loss weight is 10.0')
        self.parser.add_argument('--use_norm', type=float, default=1, help='L1 loss weight is 10.0')
        self.parser.add_argument('--use_wgan', type=float, default=0, help='use wgan-gp')
        self.parser.add_argument('--use_ragan', action='store_true', help='use ragan')
        self.parser.add_argument('--vgg', type=float, default=0, help='use perceptrual loss')
        self.parser.add_argument('--vgg_mean', action='store_true', help='substract mean in vgg loss')
        self.parser.add_argument('--vgg_choose', type=str, default='relu5_3', help='choose layer for vgg')
        self.parser.add_argument('--no_vgg_instance', action='store_true', help='vgg instance normalization')
        self.parser.add_argument('--vgg_maxpooling', action='store_true', help='normalize attention map')
        self.parser.add_argument('--IN_vgg', action='store_true', help='patch vgg individual')
        self.parser.add_argument('--fcn', type=float, default=0, help='use semantic loss')
        self.parser.add_argument('--use_avgpool', type=float, default=0, help='use perceptrual loss')
        self.parser.add_argument('--instance_norm', type=float, default=0, help='use instance normalization')
        self.parser.add_argument('--syn_norm', action='store_true', help='use synchronize batch normalization')
        self.parser.add_argument('--tanh', type=bool, default=True, help='tanh')
        self.parser.add_argument('--linear', action='store_true', help='tanh')
        self.parser.add_argument('--new_lr', action='store_true', help='tanh')
        self.parser.add_argument('--multiply', action='store_true', help='tanh')
        self.parser.add_argument('--noise', type=float, default=0, help='variance of noise')
        self.parser.add_argument('--input_linear', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--linear_add', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--latent_threshold', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--latent_norm', action='store_true', help='lieanr scaling input')
        self.parser.add_argument('--patchD', action='store_true', help='use patch discriminator')
        self.parser.add_argument('--patchD_3', type=int, default=0, help='choose the number of crop for patch discriminator')
        self.parser.add_argument('--D_P_times2', action='store_true', help='loss_D_P *= 2')
        self.parser.add_argument('--patch_vgg', action='store_true', help='use vgg loss between each patch')
        self.parser.add_argument('--hybrid_loss', action='store_true', help='use lsgan and ragan separately')
        self.parser.add_argument('--self_attention', action='store_true', help='adding attention on the input of generator')
        self.parser.add_argument('--times_residual', type=bool, default=True, help='output = input + residual*attention')
        self.parser.add_argument('--low_times', type=int, default=200, help='choose the number of crop for patch discriminator')
        self.parser.add_argument('--high_times', type=int, default=400, help='choose the number of crop for patch discriminator')
        self.parser.add_argument('--norm_attention', action='store_true', help='normalize attention map')
        self.parser.add_argument('--vary', type=int, default=1, help='use light data augmentation')
        self.parser.add_argument('--lighten', action='store_true', help='normalize attention map')
        self.parser.add_argument('--is_haze', action='store_true', help='haze image predicting')
        self.parser.add_argument('--is_re', action='store_true', help=' image predicting resize')
        self.parser.add_argument('--is_toushe', action='store_true', help=' image predicting resize')
        self.parser.add_argument('--is_ca', action='store_true', help=' image predicting resize')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        return self.opt

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.isTrain = False