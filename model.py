import os
import torch
import numpy as np
from torch.autograd import Variable
import networks

def create_model(opt):
    model = Model()
    model.initialize(opt)
    return model

def latent2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].detach().cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    image_numpy = np.maximum(image_numpy, 0)
    image_numpy = np.minimum(image_numpy, 255)
    return image_numpy.astype(imtype)

class BaseModel():
    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate():
        pass

class Model(BaseModel):
    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, 600, 400)
        self.input_B = self.Tensor(nb, opt.output_nc, 600, 400)
        self.input_C = self.Tensor(nb, opt.output_nc, 600, 400)
        self.input_C_gray = self.Tensor(nb, opt.output_nc, 600, 400)

        skip = True if opt.skip > 0 else False
        self.s_net = networks.define_s(self.gpu_ids,skip=skip, opt=opt)
        self.d_net = networks.define_d(self.gpu_ids, skip=skip, opt=opt)
        self.f_net = networks.define_f(self.gpu_ids, skip=skip, opt=opt)

        which_epoch = 'best'
        print("---model is loaded---")
        self.load_network(self.s_net, 'Att', which_epoch)
        self.load_network(self.d_net, 'Con', which_epoch)
        self.load_network(self.f_net, 'F', which_epoch)

        print('---Networks initialized ---')
        self.s_net.eval()
        self.d_net.eval()
        self.f_net.eval()

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A']
        input_B = input['B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def predict(self):
        nb = self.opt.batchSize
        size = self.opt.fineSize
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)

        self.a1 = self.d_net.forward(self.real_A)
        self.a2 = self.s_net.forward(self.real_B)
        self.output1 = self.a1 * self.real_A + self.a2 * self.real_B
        self.latent, self.output ,self.edge = self.f_net.forward(self.output1)
        output = latent2im(self.output.data)
        return output

    # get image paths
    def get_image_paths(self):
        return self.image_paths

