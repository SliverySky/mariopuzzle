import sys,os
sys.path.append(os.path.dirname(sys.path[0]))
import PIL.Image as Image
from PIL import ImageDraw
from pcg_gym.envs.MarioLevelRepairer.utils.level_process import *
IMAGES_PATH = rootpath + '/Assets/Tiles'
IMAGES_FORMAT = ['.jpg', '.JPG', '.PNG', '.png']
IMAGE_SIZE = 16

image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]

def saveLevelAsImage(level, path):
    IMAGE_ROW = len(level)
    IMAGE_COLUMN = len(level[0])
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + "/" + str(level[y - 1][x - 1]) + ".jpg").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(path + ".jpg")

def saveAndMark(level, name, S, T):
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

def save_level_as_text(level, name):
    with open(name+".txt", 'w') as f:
        f.write(arr_to_str(level))
