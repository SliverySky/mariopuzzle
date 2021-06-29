import sys,os
sys.path.append(sys.path[0]+'//..'+'//..')
os.chdir(sys.path[0])
import torch
from torch.autograd import Variable
from LevelGenerator.GAN.dcgan import Generator
from utils.level_process import *
from utils.visualization import *
from root import rootpath


def get_level(noise, to_string, name, size):
    model_to_load = name
    batch_size = 1
    image_size = 32 * size
    ngf = 64
    nz = 32
    z_dims = 10  # number different titles
    generator = Generator(nz, ngf, image_size, z_dims)
    generator.load_state_dict(torch.load(model_to_load, map_location=lambda storage, loc: storage))
    latent_vector = torch.FloatTensor(noise).view(batch_size, nz, 1, 1)
    with torch.no_grad():
        levels = generator(Variable(latent_vector))
    im = levels.data.cpu().numpy()
    im = np.argmax(im, axis=1)
    im = little_level(im[0], size)
    if to_string:
        return arr_to_str(im[0:14, 0:28])
    else:
        return im[0:14, 0:28]
def get_random_long_level():
    lvs = []
    for i in range(int(120/28)):
        lvs.append(get_level(np.random.randn(1, 32), False, './generator.pth', 1))
    lv = np.concatenate(lvs, axis=-1)
    lv = addLine(lv)
    return lv

if __name__ == '__main__':
    lvs = []
    total = 100
    select = 5
    for i in range(total):
        print('\rgenerate',i,end='')
        lv = get_random_long_level()
        cnt = calculate_broken_pipes(lv)
        lvs.append((cnt, lv))
    lvs.sort(key=lambda s:s[0], reverse=True)
    cnt_sum = 0
    print()
    for i in range(select):
        saveLevelAsImage(lvs[i][1], 'Destroyed//lv'+str(i))
        with open('Destroyed//lv'+str(i)+'.txt', 'w') as f:
            f.write(arr_to_str(lvs[i][1]))
        print('lv'+str(i)+': cnt=', str(lvs[i][0]))
        cnt_sum += lvs[i][0]
    print('avg_broken_pipe_combinations=', cnt_sum / total)
    print('Defective levels are saved in LevelGenerator//GAN//Destroyed folder.')
