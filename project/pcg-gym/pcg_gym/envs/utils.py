import time
import numpy as np
import math
import os
import PIL.Image as Image
from PIL import ImageDraw, ImageFont
def mapEntropy(mp):
    sum0 = sum(mp[e] for e in mp.keys())
    res = 0
    for e in mp.keys():
        p = mp[e]/sum0
        res -= p*math.log2(p)
    return res
def calKLFromMap(mpa, mpb, w=0.5, eps=0.001):
    result = 0
    keys = set(mpa.keys()) | set(mpb.keys())
    suma = sum([mpa[e] for e in mpa.keys()])
    sumb = sum([mpb[e] for e in mpb.keys()])
    for e in keys:
        a = ((eps + mpa[e]) / (suma + len(keys) * eps)) if (e in mpa.keys()) else (eps / (suma + len(keys) * eps));
        b = ((eps + mpb[e]) / (sumb + len(keys) * eps)) if (e in mpb.keys()) else (eps / (sumb + len(keys) * eps))
        result += w * a * math.log2(a / b) + (1 - w) * b * math.log2(b / a)
    return result
def calKL(mpa, mpb, eps=0.001):
    result = 0
    keys = set(mpa.keys()) | set(mpb.keys())
    suma = sum([mpa[e] for e in mpa.keys()])
    sumb = sum([mpb[e] for e in mpb.keys()])
    for e in keys:
        a = ((eps + mpa[e]) / (suma + len(keys) * eps)) if (e in mpa.keys()) else (eps / (suma + len(keys) * eps));
        b = ((eps + mpb[e]) / (sumb + len(keys) * eps)) if (e in mpb.keys()) else (eps / (sumb + len(keys) * eps))
        result += a * math.log2(a / b) 
    return result
def lv2Map(lv, fh=2, fw=2):
    mp={}
    h, w = lv.shape
    for i in range(h-fh+1):
        for j in range(w-fw+1):
            k = tuple((lv[i:i+fh, j:j+fw]).flatten())
            mp[k] = (mp[k]+1) if (k in mp.keys()) else 1
    return mp

def KLWithSlideWindow(lv, window, sx, nx, sy=14, ny=0):
    y, x, sh, sw = window
    _ny = min(y // sy, ny)
    _nx = min(x // sx, nx)
    mp0 = lv2Map(lv[y:y+sh, x:x+sw])
    res = 0
    for i in range(_ny+1):
        for j in range(_nx+1):
            p = lv[y-sy*i:y-sy*i+sh, x-sx*j:x-sx*j+sw]
            res += calKLFromMap(mp0, lv2Map(p))
    num = (_ny+1)*(_nx+1) 
    if num!= 0: res/=num
    return res
def saveLevelAsImage(level, path, line=0, mark={}):
    IMAGES_PATH = os.path.dirname(__file__)  + '/tiles'
    IMAGES_FORMAT = ['.jpg', '.JPG', '.PNG', '.png']
    IMAGE_SIZE = 16

    image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
                   os.path.splitext(name)[1] == item]
    IMAGE_ROW = len(level)
    IMAGE_COLUMN = len(level[0])
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + "/" + str(level[y - 1][x - 1]) + ".jpg").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    #ttf = ImageFont.load_default()
    print(os.path.dirname(os.path.abspath(__file__))+"/font.OTF")
    ttf = ImageFont.truetype(os.path.dirname(os.path.abspath(__file__))+"/font.OTF", 20)
    draw = ImageDraw.Draw(to_image)
    if line != 0:
        for i in range(IMAGE_COLUMN//line):
            if i in mark.keys():
                print('rew=',mark[i])
                if mark[str(i)+'c']>0.9:
                    c = (0,255,0)
                elif mark[str(i)+'c']>0.8:
                    c = (0,0,255)
                else:
                    c = (255,0,0)
                draw.text((i*line*IMAGE_SIZE, 2), mark[i], font=ttf, fill=c)
            for t in range(IMAGE_ROW*IMAGE_SIZE//5):
                if t%2 == 1: continue
                st = (i*line*IMAGE_SIZE,t*5)
                ft = (i*line*IMAGE_SIZE,t*5+5)
                draw.line(st + ft, fill=0x000000, width=1, joint="curve")
    lt, rt, lb, rb= (0,0), (IMAGE_COLUMN*IMAGE_SIZE, 0), (0, IMAGE_ROW* IMAGE_SIZE), (IMAGE_COLUMN*IMAGE_SIZE, IMAGE_ROW* IMAGE_SIZE)
    draw.line(lt + rt, fill=0x000000, width=3)
    draw.line(rt + rb, fill=0x000000, width=3)
    draw.line(lb + rb, fill=0x000000, width=3)
    draw.line(lt + lb, fill=0x000000, width=3)
    print(path)
    return to_image.save(path + ".jpg")
def saveLevelAsText(level, path):
    map={'X':0, 'S':1, '-':2, '?':3, 'Q':4, 'E':5,'<':6,'>':7,'[':8,']':9,'o':10,'B':11,'b':12}
    map2=['X','S','-', '?', 'Q', 'E','<','>','[',']','o','B','b']
    with open(path+".txt",'w') as f:
        for i in range(len(level)):
            str=''
            for j in range(len(level[0])):
                str+=map2[level[i][j]]
            f.write(str+'\n')
def readTextLevel(path):
    map={'X':0, 'S':1, '-':2, '?':3, 'Q':4, 'E':5,'<':6,'>':7,'[':8,']':9,'o':10,'B':11,'b':12}
    result = []
    arr = None
    with open(path) as f:
        data = f.readlines()
        h, w = len(data), len(data[0])-1
        arr = np.empty(shape=(h,w), dtype=int)
        for i in range(h):
            for j in range(w):
                arr[i][j]=map[data[i][j]]
    return arr
def saveAndMark(level, name, S, T):
    IMAGES_PATH = os.path.dirname(__file__)  + '/tiles'
    IMAGE_SIZE = 16
    IMAGE_ROW = len(level)
    IMAGE_COLUMN = len(level[0])
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + "/" + str(level[y - 1][x - 1]) + ".jpg").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
            for pos in S:
                lt = (pos[1] * 16 + 1.5, pos[0] * 16 + 1.5)
                rt = (pos[1] * 16 + 1.5, pos[0] * 16 + 16 - 1.5)
                lb = (pos[1] * 16 + 16 - 1.5, pos[0] * 16 + 1.5)
                rb = (pos[1] * 16 + 16 - 1.5, pos[0] * 16 + 16 - 1.5)
                draw = ImageDraw.Draw(to_image)
                draw.line(lt + rt, fill=0x0000FF, width=3)
                draw.line(lt + lb, fill=0x0000FF, width=3)
                draw.line(rt + rb, fill=0x0000FF, width=3)
                draw.line(lb + rb, fill=0x0000FF, width=3)
            for pos in T:
                lt = (pos[1] * 16 + 1.5, pos[0] * 16 + 1.5)
                rt = (pos[1] * 16 + 1.5, pos[0] * 16 + 16 - 1.5)
                lb = (pos[1] * 16 + 16 - 1.5, pos[0] * 16 + 1.5)
                rb = (pos[1] * 16 + 16 - 1.5, pos[0] * 16 + 16 - 1.5)
                draw = ImageDraw.Draw(to_image)
                draw.line(lt + rt, fill=(0, 0, 255), width=3)
                draw.line(lt + lb, fill=(0, 0, 255), width=3)
                draw.line(rt + rb, fill=(0, 0, 255), width=3)
                draw.line(lb + rb, fill=(0, 0, 255), width=3)
    to_image.save(name + ".jpg")
def showLevelSeg(level,name=""):
    IMAGES_PATH = os.path.dirname(__file__)  + '/tiles'
    IMAGES_FORMAT = ['.jpg', '.JPG', '.PNG', '.png']
    IMAGE_SIZE = 16
    image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
                   os.path.splitext(name)[1] == item]
    IMAGE_ROW = len(level)
    IMAGE_COLUMN = len(level[0])
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + "/" + str(level[y - 1][x - 1]) + ".jpg").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    if name!="":name="_"+name
    to_image.save("/home/cseadmin/sty/project2/tmp/"+str(time.time())+name+".jpg")
    saveLevelAsText(level,"/home/cseadmin/sty/project2/tmp/"+str(time.time())+name)
def cal_coins(lv):
    return np.sum(np.where(lv==10, 1, 0)) # O
def cal_gaps(lv):
    s=""
    for x in list(lv[13, :]):
        if x == 2:
            s += "1"
        elif x== 0:
            s += '0'
        else:
            s += '3'
    s += '0'
    return s.count('10') 
def cal_pipes(lv):
    return np.sum(np.where(lv==6, 1, 0)) # <
def cal_bullets(lv):
    return np.sum(np.where(lv==11, 1, 0)) # B
def cal_enemies(lv):
    return np.sum(np.where(lv==5, 1, 0)) # E
def cal_questions(lv):
    return np.sum(np.where(np.bitwise_or(lv==3, lv==4), 1, 0))
def cal_kl(lv):
    mp = {}
    for i in range(lv.shape[0]):
        for j in range(lv.shape[1]):
            t = lv[i][j]
            if t not in mp.keys():
                mp[t]=0
            mp[t]+=1
    s = sum([mp[x] for x in mp.keys()])
    res = 0
    for x in mp.keys():
        p = mp[x] / s
        res -= p * np.log2(p)
    return res