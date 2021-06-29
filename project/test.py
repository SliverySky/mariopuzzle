from pcg_gym.envs.utils import * 
import numpy as np
a = np.full((14,14),2)
b = np.full((14,14),0)
print(calKLFromMap(lv2Map(a), lv2Map(b)))

c = np.full((14,14),2)
c[7][7]=0
c[7][8]=0
c[8][7]=0
c[8][8]=0
print(calKLFromMap(lv2Map(a), lv2Map(c)))
