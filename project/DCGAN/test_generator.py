
import models.dcgan as dcgan
import torch, json, numpy
import random
from torch.autograd import Variable
from Utils.visualization import saveLevelAsImage
from Utils.tools import cal_kl_divergence
import matplotlib.pyplot as plt
import math
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:]
    image = numpy.zeros((height*shape[0], width*shape[1],shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img
    return image

batchSize = 10
imageSize = 32
ngf = 64
ngpu = 1
n_extra_layers = 0
z_dims = 13  # number different titles
nz = 32
generator = dcgan.DCGAN_G(imageSize, nz, z_dims, ngf, ngpu, n_extra_layers)
# generator.load_state_dict(torch.load('netG_epoch_24.pth', map_location=lambda storage, loc: storage))
generator.load_state_dict(torch.load("netG_epoch_200000_0_32.pth"))
line = []
for i in range(batchSize):
    line.append([random.uniform(-10.0, 10.0) for k in range(nz)])
print(line)
line = json.dumps(line)
lv = numpy.array(json.loads(line))
latent_vector = torch.FloatTensor(lv).view(batchSize, nz, 1, 1)
levels = generator(Variable(latent_vector, volatile=True))
im = levels.data.cpu().numpy()
im = im[:,:,:14,:28] #Cut of rest to fit the 14x28 tile dimensions
im = numpy.argmax( im, axis = 1)
#print(json.dumps(levels.data.tolist()))
print(im)
print("Saving to file ")
# im = ( plt.get_cmap('rainbow')( im/float(z_dims) ) )
#
# plt.imsave('fake_sample.png', combine_images(im) )
for i in range(len(im)):
    saveLevelAsImage(im[i],str(i))
    print(cal_kl_divergence(im[0], im[i]))