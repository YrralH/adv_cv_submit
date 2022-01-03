
import os
import sys

import numpy as np
import cv2 as cv
import torch

import lib.mcubes_pytorch.mcubes_module as mc
from lib.render import White_Renderer

import multiprocessing
class Process_MCube(multiprocessing.Process) :
    def __init__(self, queue_prev, queue_next, name='', device=torch.device('cuda:1')) :
        super().__init__()
        self.name = name
        self.device = device
        self.queue_prev = queue_prev
        self.queue_next = queue_next
        return

    def run(self) :
        print('Process ' + self.name +  ' started.')
        with torch.no_grad() :
            while(True) :
                data_in = self.queue_prev.get()

                #no to device
                #verts, faces = mc.mcubes_cuda(data_in['main']['sdf'].to(self.device), 0.5)
                sdf = data_in['main']['sdf']
                if sdf is not None :
                    if self.device is not None :
                        verts, faces = mc.mcubes_cuda(sdf.to(self.device), 0.5)
                    else :
                        verts, faces = mc.mcubes_cuda(sdf, 0.5)
                    
                    verts = (verts - 128.0) / 128.0
                    #verts[:, 2] = verts[:, 2] / 2.0
                    verts = verts.cpu()
                    faces = faces.cpu()
                else :
                    verts = None
                    faces = None


                data_main = {'verts': verts, 'faces':faces}
                data_out = {'main': data_main, 'vis':data_in['vis']}

                self.queue_next.put(data_out)
            return


class Process_MCube_and_Render(multiprocessing.Process) :
    def __init__(self, queue_prev, queue_next, name='', device=torch.device('cuda:1')) :
        super().__init__()
        self.name = name
        self.device = device
        self.queue_prev = queue_prev
        self.queue_next = queue_next
        return

    def run(self) :
        white_render = White_Renderer(self.device)

        while(True) :
            data_in = self.queue_prev.get()

            verts, faces = mc.mcubes_cuda(data_in['main']['sdf'].to(self.device), 0.5)
            verts = (verts - 128.0) / 128.0

            image = white_render.render(v=verts, f=faces)

            data_main = {'render': image}
            data_out = {'main': data_main, 'vis':data_in['vis']}

            self.queue_next.put(data_out)
        return

