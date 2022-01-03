import sys
import os
import time
import multiprocessing

import cv2 as cv
import numpy as np
import torch

from lib.streamer import Process_Streamer

from PyQt5.QtWidgets import QApplication


import PyQt5.QtGui as QtGui 
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QMainWindow, 
    QWidget, 
    QLabel, 
    QPushButton,
    QLineEdit,
    QVBoxLayout, 
    QHBoxLayout, 
    QFormLayout,
    QGridLayout
)

 
class My_MainWindow(QMainWindow):
    def __init__(self, data_queue=[]) :
        super().__init__()
        self.data_queue = data_queue
        self.declare_ui_interface()
        self.lb_img_cam.setFixedSize(360, 640)
        self.lb_img_fgr.setFixedSize(360, 640)
        self.lb_img_seg.setFixedSize(360, 640)
        self.init_ui()

        self.img_cam = None
        self.PATH_BG = './data/bg/0.png'

        self.timer = QTimer(self)

        self.timer.timeout.connect(self.update_all)
        self.bt_save_bg.clicked.connect(self.on_click_save_bg)

        self.timer.start(80)
        self.show()
        return
    
    def declare_ui_interface(self) :
        self.lb_img_cam = QLabel()
        self.lb_img_fgr = QLabel()
        self.lb_img_seg = QLabel()

        self.bt_save_bg = QPushButton()
        self.bt_test_seg = QPushButton()
        return

    def init_ui(self) :
        self.wgt_center = QWidget()
        self.setCentralWidget(self.wgt_center)

        self.hbl_main = QHBoxLayout()
        self.wgt_center.setLayout(self.hbl_main)
        self.hbl_main.addWidget(self.lb_img_cam)
        self.hbl_main.addWidget(self.lb_img_fgr)
        self.hbl_main.addWidget(self.lb_img_seg)

        self.vbl_buttons = QVBoxLayout()
        self.hbl_main.addLayout(self.vbl_buttons)
        self.vbl_buttons.addStretch()
        self.vbl_buttons.addWidget(self.bt_save_bg)
        self.vbl_buttons.addWidget(self.bt_test_seg)

        self.bt_save_bg.setText('Save as BG')
        self.bt_test_seg.setText('Test Seg')
        return

    def update_frame(self, lb_frame, img) :
        if img is not None :
            qimg_frame = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QtGui.QImage.Format_BGR888)
            qpixmap_frame = QtGui.QPixmap.fromImage(qimg_frame)
            lb_frame.setPixmap(qpixmap_frame)
        else :
            None
        return


    def update_all(self) :
        queue_in = self.data_queue[-1]
        if queue_in.empty() :
            return
        data = queue_in.get()
        self.img_cam = data
        data_show = cv.resize(data, (360, 640))
        self.update_frame(self.lb_img_cam, data_show)
        return

    def on_click_save_bg(self) :
        cv.imwrite(self.PATH_BG, self.img_cam)
        return


def set_bg() :
    app = QApplication(sys.argv)

    data1 = multiprocessing.Queue(4) #frame
    data2 = multiprocessing.Queue(4)

    stage1 = Process_Streamer(
        queue_next=data1, 
        name='streamer', 
        rotate=270, 
        fps_expect=30,
        source='webcam'
    )

    my_windows = My_MainWindow(data_queue=[data1])

    stage1.daemon = True
    stage1.start()
    sys.exit(app.exec_())
    return

if __name__ == '__main__' :
    with torch.no_grad() :
        set_bg()