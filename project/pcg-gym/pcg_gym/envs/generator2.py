from pcg_gym.envs.models.dcgan import DCGAN_G
import torch
from torch.autograd import Variable
import numpy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
class Generator():
    def __init__(self, id, index=0, gpu=False):
        os.environ["CUDA_VISIBLE_DEVICES"]=str(index)
        w=14
        h=14
        path = os.path.dirname(__file__) + "//models//" + str(14)+"_"+str(14)+".pth"
        imageSize = 32
        ngf = 64
        ngpu = 1
        n_extra_layers = 0
        z_dims = 13  # number different titles
        self.nz = nz = 32
        self.h = h
        self.w = w
        self.generator = DCGAN_G(imageSize, nz, z_dims, ngf, ngpu, n_extra_layers)
        self.gpu = gpu
        if gpu:
            self.generator.load_state_dict(torch.load(path))
            self.generator = self.generator.cuda()
        else:
            self.generator.load_state_dict(torch.load(path, map_location='cpu'))
    def generate(self, noise):
        #print(noise.shape)
        latent_vector = torch.FloatTensor(noise).view(1, self.nz, 1, 1)
        if self.gpu:
            with torch.no_grad():
                levels = self.generator(Variable(latent_vector).cuda())
        else: 
            with torch.no_grad():
                levels = self.generator(Variable(latent_vector))
        im = levels.data.cpu().numpy()
        im = im[:, :, :self.h, :self.w]  # Cut of rest to fit the 14x28 tile dimensions
        im = numpy.argmax(im, axis=1)
        return im[0]