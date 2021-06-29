from pcg_gym.envs.search_latent import *
from pcg_gym.envs.utils import *
from pcg_gym.envs.generator import *
import os
root = os.path.abspath(os.path.dirname(__file__))
model_path = root + "//models//14_7.pth"
generator = Generator(model_path, 14, 14)
#res = generator.generate(np.array([0]*64).resize(2,32))
#print(res)
searcher = Searcher(generator)
names = ['1-1']
for name in names:
    lv = readTextLevel(root+"//latent//"+name+".txt")
    saveLevelAsImage(lv, root+"//latent//"+name+"_out")
    latent = searcher.find_latent(lv)
    in_lv = generator.generate(latent)
    saveLevelAsImage(in_lv, root+"//latent//"+name+"_in")
    with open(root+"//vector.txt", "a") as f:
        f.write(name+":{}\n".format(latent))