import sys,os
sys.path.append(sys.path[0]+'//..'+'//..')
os.chdir(sys.path[0])
from utils.visualization import *


if __name__ == '__main__':
    num = 10
    lv_str = ''
    with open(rootpath + "//LevelText//pipes.txt") as f:
        for i in f.readlines():
            lv_str += i
    lv = numpy_level(lv_str)
    for i in range(num):
        print('generate ',i)
        new_lv = random_destroy(lv)
        with open('lv'+str(i)+'.txt', 'w') as f:
            f.write(arr_to_str(new_lv))
        saveLevelAsImage(new_lv, 'lv' + str(i))
        print('Defective levels are saved in LevelGenerator//RandomDestroyed folder.')
