
import os

import time
import cv2 as cv
import numpy as np
import torch
from PyQt5.QtCore import QThread

class Q_Refresher(QThread):
    def __init__(self, main_window, list_queue) :
        super().__init__()
        self.main_window = main_window
        self.list_queue = list_queue
        self.queue_out = self.list_queue[-1]
        self.status_azim_circle = False
        return

    def run(self) :
        count = 0
        time_fps_start = time.time()
        list_qsize = []
        for i in range(len(self.list_queue)) :
            list_qsize.append(0)
            
        print('Q_Refresher started running.')

        while(True) :
            t_s = time.time()

            if self.status_azim_circle :
                self.main_window.azim_add()

            data_in = self.queue_out.get()
            img_geo_tensor = data_in['main']['render'] * 255
            img_geo = img_geo_tensor.to(torch.uint8).cpu().numpy().astype(np.uint8)
            img_seg = data_in['vis']['seg']
            img_det = data_in['vis']['det']

            for i in range(len(self.list_queue)) :
                list_qsize[i] = self.list_queue[i].qsize()

            self.main_window.update_qsize(list_qsize)
            self.main_window.update_all_frame(
                res_geo=img_geo, 
                res_tex=None, 
                vis_det=img_det, 
                vis_seg=img_seg
            )

            count = count + 1
            if count == 10 :
                t_fps_end = time.time()
                fps = count / (t_fps_end - time_fps_start)
                self.main_window.update_fps(fps)
                time_fps_start = time.time()
                count = 0
            t_e = time.time()
            #print(t_e - t_s)
