import os, sys
import shutil
os.chdir(sys.path[0])

def del_file(path):
    shutil.rmtree(path)
    os.mkdir(path)


if __name__ == '__main__':
    del_file('result')
    os.mkdir('result//figure')
    os.mkdir('result//json')
    os.mkdir('result//txt')
