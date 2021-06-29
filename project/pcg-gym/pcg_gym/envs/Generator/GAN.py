import pcg_gym.envs.models.dcgan as dcgan
import torch
from torch.autograd import Variable
import numpy as np
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
rootpath = os.path.abspath(os.path.dirname(__file__))
import time
class GAN():
    def __init__(self):
        path = "/home/cseadmin/sty/project2/pcg-gym/pcg_gym/envs" + "//models//" + str(14) + "_" + str(14) + ".pth"
        gpu = True
        imageSize = 32
        ngf = 64
        ngpu = 1
        n_extra_layers = 0
        z_dims = 13  # number different titles
        self.nz = nz = 32
        self.h = 14
        self.w = 14
        self.generator = dcgan.DCGAN_G(imageSize, nz, z_dims, ngf, ngpu, n_extra_layers)
        self.gpu = gpu
        if gpu:
            self.generator.load_state_dict(torch.load(path))
            self.generator = self.generator.cuda()
        else:
            self.generator.load_state_dict(torch.load(path, map_location='cpu'))
    def generate(self, noises):
        latent_vector = torch.FloatTensor(noises).view(-1, self.nz, 1, 1)
        if self.gpu:
            with torch.no_grad():
                levels = self.generator(Variable(latent_vector).cuda())
        else:
            with torch.no_grad():
                levels = self.generator(Variable(latent_vector, volatile=True))
        im = levels.data.cpu().numpy()
        im = im[:, :, :self.h, :self.w]  # Cut of rest to fit the 14x28 tile dimensions
        im = np.argmax(im, axis=1)
        return im
while True:
    name = []
    data = []
    out_name = []
    for root, dirs, files in os.walk(rootpath+"//input"):
        for file in files:
            path = os.path.abspath(os.path.join(root, file))
            if os.path.getsize(path)>0:
                with open(path) as f:
                    data.append(json.load(f))
                    name.append(path)
                    out_name.append(rootpath+"//output//"+file)
    if len(name)==0:continue
    noises = np.array(data)
    generator = GAN()
    levels = generator.generate(noises)
    assert len(name)==len(data)
    assert len(name)==len(out_name)
    for i in range(len(name)):
        os.remove(name[i])
        with open(out_name[i],"w") as f:
            json.dump(levels[i].tolist(), f)
    time.sleep(0.01)