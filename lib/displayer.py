import os
import sys
import time
import multiprocessing

#extenal lib
import numpy as np
import cv2 as cv

#pytorch
import torch
import torchvision




class Process_Displayer(multiprocessing.Process) :
    def __init__(self, queue_prev, main_window, name='') :
        super().__init__()
        self.name = name
        self.queue_prev = queue_prev
        self.main_window = main_window
        return

    def run(self) :
        print('Process ' + self.name +  ' started.')
        i = 0

        while(True) :
            t_s = time.time()
            print('------begin------')
            data_in = self.queue_prev.get()

            img_geo_tensor = data_in['main']['render'] * 255
            img_geo = img_geo_tensor.to(torch.uint8).cpu().numpy().astype(np.uint8)
            #print(img_geo)
            #img_tex = None
            img_vis = data_in['vis']['seg']
            bbox = data_in['vis']['bbox']

            self.main_window.update_all(
                res_geo=img_geo, 
                res_tex=None, 
                vis_det=None, 
                vis_seg=None
            )

            
            #print(data1.qsize())
            #print(data2.qsize())
            #print(data3.qsize())
            #print(data4.qsize())
            #print(data5.qsize())
            #cv.imwrite(
            #    os.path.join('./' ,'results', 'out_frame', str(i) + '.png'), 
            #    (image_geo.cpu().numpy() * 255).astype(np.uint8)
            #)
            #i = i + 1
            t_e = time.time()
            print(t_e - t_s)
        return