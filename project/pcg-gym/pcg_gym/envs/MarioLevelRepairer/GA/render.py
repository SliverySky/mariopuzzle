import sys,os
sys.path.append(os.path.dirname(sys.path[0]))
import pygame
from pygame import display, image, time, transform
from pygame.font import SysFont
from root import rootpath
import os


class Renderer:
    path = rootpath + '/GA/result/figure/'
    size = (1440, 192)

    def __init__(self, surface):
        self.surface = surface
        self.status = 0 # 0: auto; 1: manul
        self.iter = 0
        self.ftw = 120
        self.font = SysFont('Comic Sans MS', 32)
        self.draw_help()

    def on_space(self):
        if self.status == 0:
            self.status = 1
            self.ftw = 120
        else:
            self.status = 0

    def on_left(self):
        if self.status == 1 and self.iter > 0:
            self.iter -= 1

    def on_right(self):
        if self.status == 1 and self.iter < Renderer.get_file_num()-1:
            self.iter += 1

    def draw_help(self):
        if self.status == 1:
            text_surface = self.font.render('Iteration' + str(self.iter)+"(>>)",
                                            True, (0, 0, 0))
        else:
            text_surface = self.font.render('Iteration' + str(self.iter)+"(||)",
                                            True, (0, 0, 0))

        self.surface.blit(text_surface, [0, 0])

    def draw_level(self):
        img = image.load(Renderer.path + 'iteration%d.jpg' % self.iter)
        img = transform.scale(img, Renderer.size)
        self.surface.blit(img, [0, 0])
        if self.status == 0:
            if self.ftw > 0:
                self.ftw -= 1
                return
            if self.iter < Renderer.get_file_num()-1:
                self.iter += 1
                self.ftw += 120

    def update(self):
        self.draw_level()
        self.draw_help()

    @staticmethod
    def get_file_num():
        _, _, files = os.walk(Renderer.path).__next__()
        return len(files)


if __name__ == '__main__':
    pygame.init()
    screen = display.set_mode(Renderer.size)
    screen.fill((255,255,255))
    display.flip()
    renderer = Renderer(screen)
    clk = time.Clock()

    while True:
        clk.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    renderer.on_space()
                elif event.key == pygame.K_LEFT:
                    renderer.on_left()
                elif event.key == pygame.K_RIGHT:
                    renderer.on_right()
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
        renderer.update()
        display.flip()
