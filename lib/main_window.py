import os
import time

import cv2 as cv
import numpy as np
import torch


import PyQt5.QtGui as QtGui 
import PyQt5.QtCore as QtCore
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
    
    def __init__(self, list_queue=[], list_forbidden_city=[], render_controller=None, streamer_controller=None) :
        super().__init__()
        self.path_save_base = '/mnt/data/hj/c920pro/wild/'
        self.str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.path_save_folder = os.path.join(self.path_save_base, self.str_time)
        print('all seq saving in ' + self.path_save_folder + '.')
        os.makedirs(self.path_save_folder, exist_ok=True)

        self.render_controller = render_controller
        self.streamer_controller = streamer_controller
        self.list_queue = list_queue
        self.list_forbidden_city = list_forbidden_city
        self.list_qsize = []
        for i in range(len(self.list_queue)) :
            self.list_qsize.append(0)

        self.declare_const_cam_para()
        self.declare_ui(self.list_queue)
        

        self.dist = 3.2
        self.elev = 0.0
        self.azim = 0.0 
        self.status_pause = False

        self.image_size = 650
        #-------------pre set fixed label size-------------
        self.lb_img_det.setFixedSize(300, 411)
        self.lb_img_seg.setFixedSize(300, 411)
        self.lb_img_geo.setFixedSize(self.image_size, self.image_size)
        #self.lb_img_tex.setFixedSize(512, 512)
        #-------------pre set fixed label size end-------------
        
        self.list_lb_cam_value[0].setText('%.3f' % self.dist)
        self.list_lb_cam_value[1].setText('%.3f' % self.elev)
        self.list_lb_cam_value[2].setText('%.3f' % self.azim)

        self.list_bt_cam_cir[0].setEnabled(False)
        #self.list_bt_cam_cir[1].setEnabled(False)
        #self.list_bt_cam_cir[2].setEnabled(False)

        self.initUI()

        self.status_slightly_circle = False
        self.status_azim_circle = False
        self.status_render_normal = False
        #self.q_refresher =  Q_Refresher(main_window=self, list_queue=list_queue)
        #self.q_refresher.start()

        self.count_sligtly_circle = 0
        self.count_main = 0
        self.count_frame = 0

        self.flag_auto_slight_move = False
        self.flag_auto_slight_moving = False
        self.count_auto_slight_move = 0

        self.flag_save_input = False
        self.count_saving_folder = 0
        self.count_saving_file = 0
        self.flag_saving_continous = False
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_all)
    
        
        self.list_bt_cam_add[0].clicked.connect(self.on_click_dist_add)
        self.list_bt_cam_sub[0].clicked.connect(self.on_click_dist_sub)
        self.list_bt_cam_add[1].clicked.connect(self.on_click_elev_add)
        self.list_bt_cam_sub[1].clicked.connect(self.on_click_elev_sub)
        self.list_bt_cam_add[2].clicked.connect(self.on_click_azim_add)
        self.list_bt_cam_sub[2].clicked.connect(self.on_click_azim_sub)
        self.list_bt_cam_cir[1].clicked.connect(self.on_click_slightly_circle)
        self.list_bt_cam_cir[2].clicked.connect(self.on_click_elev_circle)
        self.bt_cam_set.clicked.connect(self.on_click_set_cam)
        self.bt_cam_reset.clicked.connect(self.on_click_reset_cam)
        self.bt_auto_slight_move.clicked.connect(self.on_click_auto_slight_move)
        self.bt_save_input.clicked.connect(self.on_click_save_input)
        self.bt_normal.clicked.connect(self.on_click_normal)
        self.bt_pause.clicked.connect(self.on_click_pause)
        self.bt_save_obj.clicked.connect(self.on_click_save_obj)
        self.bt_to_animation.clicked.connect(self.on_click_to_annimation)

        self.timer.start(50)
        return

    def __del__(self) :
        for i in self.list_forbidden_city :
            i.terminate()

    
    def declare_const_cam_para(self) :
        #cam para
        self.DEFAULT_DIST = 3.0
        self.DEFAULT_ELEV = 0.0
        self.DEFAULT_AZIM = 0.0

        self.MAX_DIST = 15.0
        self.MIN_DIST = 0.1
        self.MAX_ELEV = 90.1
        self.MIN_ELEV = -90.1
        self.MAX_AZIM = 360.0
        self.MIN_AZIM = -0.1

        self.STRIDE_DIST = 0.15
        self.STRIDE_ELEV = 10.0
        self.STRIDE_AZIM = 15.0
        return
        

    def declare_ui(self, list_queue) :
         #center
        self.wgt_center = QWidget()
        self.vbl_main = QVBoxLayout()

        #upper of vhl_main
        self.hbl_0 = QHBoxLayout()

            #left of hbl_0
        self.vbl_0_0 = QVBoxLayout()
        self.lb_img_det = QLabel() #interface
        self.lb_img_seg = QLabel() #interface
        self.lb_tag_det = QLabel()
        self.lb_tag_seg = QLabel()
        self.fl_prob = QFormLayout()
        self.lb_tag_prob = QLabel()
        self.lb_value_prob = QLabel() #interface

            #right of hbl_0
        self.vbl_0_1 = QVBoxLayout()
        self.hbl_0_1_0 = QHBoxLayout()
        self.gl_cam_0_1_1 = QGridLayout()
                #two resualt hbl_0_1_0
        self.vbl_img_geo = QVBoxLayout()
        self.vbl_img_tex = QVBoxLayout()
        self.lb_img_geo = QLabel() #interface
        self.lb_img_tex = QLabel() #interface
        self.lb_tag_geo = QLabel()
        self.lb_tag_tex = QLabel()
                #control cam para gl_cam_0_1_1
        '''
        +---------------------------------+
        |dist|value|-|+|循环|input|       |
        |elev|value|-|+|循环|input|defualt|
        |azim|value|-|+|循环|input|set    |
        |    |     | | |    |     |      |
        |    |     | | |暂停|save |anima  |
        +---------------------------------+
        '''
        self.list_lb_cam_tag = []
        self.list_lb_cam_value = [] #interface
        self.list_bt_cam_sub = [] #interface
        self.list_bt_cam_add = [] #interface
        self.list_bt_cam_cir = [] #interface
        self.list_le_cam_input = [] #interface
        self.bt_cam_reset = QPushButton() #interface
        self.bt_cam_set = QPushButton() #interface
        for i in range(3) :
            lb_cam_tag = QLabel()
            lb_cam_value = QLabel()
            bt_cam_sub = QPushButton()
            bt_cam_add = QPushButton()
            bt_cam_cir = QPushButton()
            le_cam_input = QLineEdit()
            self.list_lb_cam_tag.append(lb_cam_tag)
            self.list_lb_cam_value.append(lb_cam_value)
            self.list_bt_cam_sub.append(bt_cam_sub)
            self.list_bt_cam_add.append(bt_cam_add)
            self.list_bt_cam_cir.append(bt_cam_cir)
            self.list_le_cam_input.append(le_cam_input)
        self.bt_auto_slight_move = QPushButton()
        self.bt_pause = QPushButton()
        self.bt_save_input = QPushButton()
        self.bt_save_obj = QPushButton()
        self.bt_to_animation = QPushButton()
        self.bt_normal = QPushButton()
        
        
        #bottom of vhl_main
        self.hbl_1 = QHBoxLayout()
            #left of hbl_1
        #self.fl_qsize = QFormLayout()
        self.list_lb_tag_qsize = [] #adapt to input
        self.list_lb_value_qsize = [] #adapt to input, interface
        for i in range(len(list_queue)) :
            lb_tag_qsize = QLabel()
            lb_value_qsize = QLabel()
            self.list_lb_tag_qsize.append(lb_tag_qsize)
            self.list_lb_value_qsize.append(lb_value_qsize)
        #self.fl_fps = QFormLayout()
        self.lb_value_fps = QLabel()
        self.lb_tag_fps = QLabel()
        return


        
    def initUI(self) :
        font1 = QtGui.QFont()
        font2 = QtGui.QFont()
        font1.setFamily('Arial')
        font1.setBold(True)
        font1.setPointSize(18) 
        font2.setFamily('Arial')
        font2.setBold(True)
        font2.setPointSize(24) 



        self.setCentralWidget(self.wgt_center)
        self.wgt_center.setLayout(self.vbl_main)
        self.vbl_main.addLayout(self.hbl_0)
        self.vbl_main.addLayout(self.hbl_1)
        
        #upper of vhl_main
        self.hbl_0.addLayout(self.vbl_0_0)
        self.hbl_0.addLayout(self.vbl_0_1)

            #left of hbl_0
        self.vbl_0_0.addWidget(self.lb_img_det)
        self.vbl_0_0.addWidget(self.lb_tag_det)
        self.vbl_0_0.addLayout(self.fl_prob)
        self.vbl_0_0.addWidget(self.lb_img_seg)
        self.vbl_0_0.addWidget(self.lb_tag_seg)
        #self.lb_tag_det.setText('检测结果')
        self.lb_tag_det.setFont(font1)
        self.lb_tag_seg.setFont(font1)
        self.lb_tag_det.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_tag_seg.setAlignment(QtCore.Qt.AlignCenter)
        self.lb_tag_det.setText('输入流')
        self.lb_tag_seg.setText('分割结果')
        self.fl_prob.addRow(self.lb_tag_prob, self.lb_value_prob)
        #self.lb_tag_prob.setText('检测框概率')

        '''
        self.vbl_0_0.addLayout(self.fl_qsize)
        self.vbl_0_0.addLayout(self.fl_fps)
        for i in range(len(self.list_lb_tag_qsize)) :
            self.fl_fps.addRow(self.list_lb_tag_qsize[i], self.list_lb_value_qsize[i])
            self.list_lb_tag_qsize[i].setText('数据队列' + str(i) + '现有元素：')
        self.fl_fps.addRow(self.lb_tag_fps, self.lb_value_fps)
        self.lb_tag_fps.setText('fps:')
        '''

        

                #control cam para gl_cam_0_1_1
        self.gl_cam_0_1_1.setColumnStretch(3, 1)
        for i in range(3) :
            self.gl_cam_0_1_1.addWidget(self.list_lb_cam_tag[i], i, 4)
            self.gl_cam_0_1_1.addWidget(self.list_lb_cam_value[i], i, 5)
            self.gl_cam_0_1_1.addWidget(self.list_bt_cam_sub[i], i, 6)
            self.gl_cam_0_1_1.addWidget(self.list_bt_cam_add[i], i, 7)
            self.gl_cam_0_1_1.addWidget(self.list_bt_cam_cir[i], i, 8)
            self.gl_cam_0_1_1.addWidget(self.list_le_cam_input[i], i, 9)
            self.list_bt_cam_sub[i].setText('-')
            self.list_bt_cam_add[i].setText('+')
        self.list_bt_cam_cir[1].setText('微动')
        self.list_bt_cam_cir[2].setText('循环')
        self.list_lb_cam_tag[0].setText('距离')
        self.list_lb_cam_tag[1].setText('俯仰角')
        self.list_lb_cam_tag[2].setText('偏航角')
        self.gl_cam_0_1_1.addWidget(self.bt_cam_reset, 1, 10)
        self.gl_cam_0_1_1.addWidget(self.bt_cam_set, 2, 10)
        self.bt_cam_reset.setText('恢复默认')
        self.bt_cam_set.setText('设置')

        self.gl_cam_0_1_1.setRowStretch(7, 1)

        index_last_row = 0
        #DISABLE FPS and DataQueue
        for i in range(len(self.list_lb_tag_qsize)) :
            #self.gl_cam_0_1_1.addWidget(self.list_lb_tag_qsize[i], index_last_row + i, 0)
            #self.gl_cam_0_1_1.addWidget(self.list_lb_value_qsize[i], index_last_row + i, 1)
            self.list_lb_tag_qsize[i].setText('数据队列' + str(i + 1) + '现有元素：')
        index_last_row = index_last_row + len(self.list_lb_tag_qsize)
        #self.gl_cam_0_1_1.addWidget(self.lb_tag_fps, index_last_row, 0)
        #self.gl_cam_0_1_1.addWidget(self.lb_value_fps, index_last_row, 1)
        self.lb_tag_fps.setText('fps:')
        index_last_row = index_last_row + 1

        
        self.gl_cam_0_1_1.addWidget(self.bt_auto_slight_move, index_last_row, 5, 2, 1)
        self.gl_cam_0_1_1.addWidget(self.bt_save_input, index_last_row, 6, 2, 1)
        self.gl_cam_0_1_1.addWidget(self.bt_normal, index_last_row, 7, 2, 1)
        self.gl_cam_0_1_1.addWidget(self.bt_pause, index_last_row, 8, 2, 1)
        self.gl_cam_0_1_1.addWidget(self.bt_save_obj, index_last_row, 9, 2, 1)
        self.gl_cam_0_1_1.addWidget(self.bt_to_animation, index_last_row, 10, 2, 1)
        self.bt_auto_slight_move.setText('自动微动')
        self.bt_save_input.setText('采集')
        self.bt_normal.setText('渲染法线')
        self.bt_pause.setText('暂停')
        self.bt_save_obj.setText('保存三维模型')
        self.bt_to_animation.setText('转到Animation')

            #right of hbl_0
        self.vbl_0_1.addLayout(self.hbl_0_1_0)
        self.vbl_0_1.addLayout(self.gl_cam_0_1_1)
                #two resualt
        self.hbl_0_1_0.addLayout(self.vbl_img_geo)
        self.hbl_0_1_0.addLayout(self.vbl_img_tex)
        self.vbl_img_geo.addWidget(self.lb_img_geo)
        self.vbl_img_geo.addWidget(self.lb_tag_geo)
        self.vbl_img_tex.addWidget(self.lb_img_tex)
        self.vbl_img_tex.addWidget(self.lb_tag_tex)
        self.lb_tag_geo.setFont(font2)
        self.lb_tag_geo.setText('重建的动态三维模型 实时采集中...')
        self.lb_tag_geo.setAlignment(QtCore.Qt.AlignCenter)
        #self.lb_tag_tex.setText('重建结果：纹理')

        #bottom of vhl_main
        #self.hbl_1.addLayout(self.xxx)
        #left of hbl_1
        #right of hbl_1

        self.setWindowTitle('Monitor')    
        self.show()
        return

    
    def on_click_pause(self) :
        print('on_click_pause')
        if not self.status_pause :
            self.bt_pause.setText('恢复')
            control_pack = {'flag_para_change':False, 'flag_save_obj':False, 'flag_to_animation':False, 'flag_render_normal_change':False, 
                            'flag_pause_change':True, 'para':None}
            #self.streamer_controller.put(False)
            self.render_controller.put(control_pack)
            self.status_pause = True
            self.lb_tag_geo.setText('重建的动态三维模型 静态展示中...')
        else :
            self.bt_pause.setText('暂停')
            control_pack = {'flag_para_change':False, 'flag_save_obj':False, 'flag_to_animation':False, 'flag_render_normal_change':False, 
                            'flag_pause_change':True, 'para':None}
            #self.streamer_controller.put(False)
            self.render_controller.put(control_pack)
            self.status_pause = False
            self.lb_tag_geo.setText('重建的动态三维模型 实时采集中...')
        return

    def on_click_save_obj(self) :
        print('on_click_save_obj')
        control_pack = {'flag_para_change':False, 'flag_save_obj':True, 'flag_to_animation':False, 'flag_render_normal_change':False, 'para':None}
        self.render_controller.put(control_pack)
        return

    def on_click_to_annimation(self) :
        print('on_click_to_annimation')
        control_pack = {'flag_para_change':False, 'flag_save_obj':False, 'flag_to_animation':True, 'flag_render_normal_change':False, 'para':None}
        self.render_controller.put(control_pack)
        return
    
    def on_click_normal(self) :
        print('on_click_normal')
        if not self.status_render_normal :
            control_pack = {'flag_para_change':False, 'flag_save_obj':False, 'flag_to_animation':False, 'flag_render_normal_change':True, 'para':None}
            self.render_controller.put(control_pack)
            self.bt_normal.setText('取消法线')
            self.status_render_normal = True
        else :
            control_pack = {'flag_para_change':False, 'flag_save_obj':False, 'flag_to_animation':False, 'flag_render_normal_change':True, 'para':None}
            self.render_controller.put(control_pack)
            self.bt_normal.setText('渲染法线')
            self.status_render_normal = False
        return

    def on_click_dist_add(self) :
        dist_tmp = self.dist + self.STRIDE_DIST
        if dist_tmp <= self.MAX_DIST :
            self.dist = dist_tmp
            self.update_cam_para()
        return
    
    def on_click_dist_sub(self) :
        dist_tmp = self.dist - self.STRIDE_DIST
        if dist_tmp >= self.MIN_DIST :
            self.dist = dist_tmp
            self.update_cam_para()
        return

    def on_click_elev_add(self) :
        elev_tmp = self.elev + self.STRIDE_ELEV
        if elev_tmp <= self.MAX_ELEV :
            self.elev = elev_tmp
            self.update_cam_para()
        return
    
    def on_click_elev_sub(self) :
        elev_tmp = self.elev - self.STRIDE_ELEV
        if elev_tmp >= self.MIN_ELEV :
            self.elev = elev_tmp
            self.update_cam_para()
        return
    
    def on_click_azim_add(self) :
        azim_tmp = (self.azim + self.STRIDE_AZIM) % 360
        if azim_tmp <= self.MAX_AZIM :
            self.azim = azim_tmp
            self.update_cam_para()
        return
    
    def on_click_azim_sub(self) :
        azim_tmp = (self.azim - self.STRIDE_AZIM) % 360
        if azim_tmp >= self.MIN_AZIM :
            self.azim = azim_tmp
            self.update_cam_para()
        return

    def on_click_set_cam(self) :
        print('on_click_set_cam')
        try :
            dist = float(self.list_le_cam_input[0].text())
            elev = float(self.list_le_cam_input[1].text())
            azim = float(self.list_le_cam_input[2].text())

            #print('on_click_set_cam', dist, elev, azim)

            if  self.MIN_DIST <= dist and dist <= self.MAX_DIST and \
                self.MIN_ELEV <= elev and elev <= self.MAX_ELEV and \
                self.MIN_AZIM <= azim and azim <= self.MAX_AZIM :
                self.dist = dist
                self.elev = elev
                self.azim = azim
                self.update_cam_para()
        except :
            print('invalid cam para')
        return
    
    def on_click_reset_cam(self) :
        print('on_click_reset_cam')
        self.dist = self.DEFAULT_DIST
        self.elev = self.DEFAULT_ELEV
        self.azim = self.DEFAULT_AZIM
        self.update_cam_para()
        return

    def on_click_slightly_circle(self) :
        print('on_click_slightly_circle')
        if not self.status_slightly_circle :
            for i in range(3) :
                self.list_bt_cam_add[i].setEnabled(False)
                self.list_bt_cam_sub[i].setEnabled(False)
            self.list_bt_cam_cir[2].setEnabled(False)
            self.bt_cam_set.setEnabled(False)
            self.bt_cam_reset.setEnabled(False)
            self.count_sligtly_circle = 0
            self.status_slightly_circle = True
        else :
            for i in range(3) :
                self.list_bt_cam_add[i].setEnabled(True)
                self.list_bt_cam_sub[i].setEnabled(True)
            self.list_bt_cam_cir[2].setEnabled(True)
            self.bt_cam_set.setEnabled(True)
            self.bt_cam_reset.setEnabled(True)
            self.status_slightly_circle = False

    
    def on_click_elev_circle(self) :
        print('on_click_elev_circle')
        if not self.status_azim_circle :
            for i in range(3) :
                self.list_bt_cam_add[i].setEnabled(False)
                self.list_bt_cam_sub[i].setEnabled(False)
            self.list_bt_cam_cir[1].setEnabled(False)
            self.bt_cam_set.setEnabled(False)
            self.bt_cam_reset.setEnabled(False)
            self.status_azim_circle = True
        else :
            for i in range(3) :
                self.list_bt_cam_add[i].setEnabled(True)
                self.list_bt_cam_sub[i].setEnabled(True)
            self.list_bt_cam_cir[1].setEnabled(True)
            self.bt_cam_set.setEnabled(True)
            self.bt_cam_reset.setEnabled(True)
            self.status_azim_circle = False

    def azim_elev_add(self) :
        self.elev = self.elev + 4
        self.azim = (self.azim + 6) % 360
        self.update_cam_para()
        return

    def azim_add(self) :
        self.azim = (self.azim + 6) % 360
        self.update_cam_para()
        return

    def azim_sub(self) :
        azim_tmp = (self.azim - 6) % 360
        if azim_tmp <= self.MAX_AZIM :
            self.azim = azim_tmp
            self.update_cam_para()
        return
    
    def elev_add(self) :
        self.elev = self.elev + 4
        self.update_cam_para()
        return

    def elev_sub(self) :
        self.elev = self.elev - 4
        self.update_cam_para()
        return

    def update_cam_para(self) :
        self.list_lb_cam_value[0].setText('%.3f' % self.dist)
        self.list_lb_cam_value[1].setText('%.3f' % self.elev)
        self.list_lb_cam_value[2].setText('%.3f' % self.azim)
        #print('update_cam_para called', self.dist, self.azim, self.elev)
        #give para to Process:Render through pipe
        control_pack = {'flag_para_change':True, 'para':(self.dist, self.azim, self.elev)}
        self.render_controller.put(control_pack)
        return


    def update_frame(self, lb_frame, img) :
        if img is not None :
            qimg_frame = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1]*3, QtGui.QImage.Format_RGB888)
            #print('img.shape', img.shape)
            #print('bytesPerLine', qimg_frame.bytesPerLine())
            #print('qimg_frame', qimg_frame.size(), qimg_frame.height(), qimg_frame.width())
            qpixmap_frame = QtGui.QPixmap.fromImage(qimg_frame)
            #print('qpixmap_frame', qpixmap_frame.size(), qpixmap_frame.height(), qpixmap_frame.width())
            lb_frame.setPixmap(qpixmap_frame)
        else :
            #print('update_frame:, img is None'))
            None
        return

    def update_frame_blk(self, lb_frame, img) :
        if img is not None :
            qimg_frame = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QtGui.QImage.Format_Grayscale8)
            qpixmap_frame = QtGui.QPixmap.fromImage(qimg_frame)
            lb_frame.setPixmap(qpixmap_frame)
        else :
            #print('update_frame:, img is None'))
            None
        return

    def update_all_frame(self, res_geo, res_tex, vis_det, vis_seg) :
        '''
        called by refresher
        effect followings
        self.lb_img_geo
        self.lb_img_tex
        self.lb_img_seg
        self.lb_img_det
        '''
        self.update_frame(self.lb_img_geo, res_geo)
        self.update_frame(self.lb_img_tex, res_tex)
        self.update_frame(self.lb_img_det, vis_det)
        self.update_frame_blk(self.lb_img_seg, vis_seg)
        return

    def update_list_lb(self, list_lb, list_value) :
        for i in range(len(list_lb)) :
            list_lb[i].setText(str(list_value[i]))

    def update_qsize(self) :
        '''
        called by refresher
        '''
        self.update_list_lb(self.list_lb_value_qsize, self.list_qsize)
    
    def update_fps(self, value) :
        self.lb_value_fps.setText('%.3f' % value)

    def slightly_circle(self, count) :
        count_inst = (count % (22 * 4)) // 4
        if count_inst in [1] :
            self.elev_add()
        elif count_inst in [5, 6] :
            self.elev_sub()
        elif count_inst in [10] :
            self.azim_elev_add()
        elif count_inst in [14, 15] :
            self.azim_sub()
        elif count_inst in [19] :
            self.azim_add()

    def saving_original_input(self, img_saving, img_mask_saving) :
        img_saving = (img_saving.permute(1, 2, 0) * 255).to(torch.uint8).numpy().astype(np.uint8)
        img_saving = cv.cvtColor(img_saving, cv.COLOR_RGB2BGR)
        img_mask_saving = (img_mask_saving.permute(1, 2, 0) * 255).to(torch.uint8).numpy().astype(np.uint8)
        img_mask_saving = cv.cvtColor(img_mask_saving, cv.COLOR_RGB2BGR)

        if self.flag_saving_continous :
            name_folder = str(self.count_saving_folder).zfill(4)
        else :
            self.count_saving_folder = self.count_saving_folder + 1
            name_folder = str(self.count_saving_folder).zfill(4)
            path_new_folder = os.path.join(self.path_save_folder, name_folder)
            print('making new seq dir: ' + path_new_folder + ' ...')
            os.makedirs(path_new_folder, exist_ok=True)
            self.count_saving_file = 0
        self.count_saving_file = self.count_saving_file + 1
        name_img = str(self.count_saving_file).zfill(6) + '.png'
        name_img_mask = str(self.count_saving_file).zfill(6) + '_mask.png'

        path_save_img = os.path.join(self.path_save_folder, name_folder, name_img)
        cv.imwrite(path_save_img, img_saving)
        path_save_img_mask = os.path.join(self.path_save_folder, name_folder, name_img_mask)
        cv.imwrite(path_save_img_mask, img_mask_saving)
        return

    def on_click_save_input(self) :
        print('on_click_save_input')
        if self.flag_save_input :
            self.flag_save_input = False
            print('Mainwindow:self.flag_save_input', self.flag_save_input)
        else :
            self.flag_save_input = True
            print('Mainwindow:self.flag_save_input', self.flag_save_input)
        return 

    def on_click_auto_slight_move(self) :
        print('on_click_auto_slight_move')
        if self.flag_auto_slight_move :
            self.flag_auto_slight_move = False
            self.bt_auto_slight_move.setText('自动微动')
        else :
            self.flag_auto_slight_move = True
            self.bt_auto_slight_move.setText('取消自动微动')
        return

    def update_all(self) :
        self.count_main = self.count_main + 1
        queue_out = self.list_queue[-1]
        if not queue_out.empty() :
            if self.flag_auto_slight_move :
                self.count_auto_slight_move = self.count_auto_slight_move + 1
                if not self.flag_auto_slight_moving :
                    if self.count_auto_slight_move > 50 :
                        self.on_click_pause()
                        self.on_click_slightly_circle()
                        self.count_auto_slight_move = 0
                        self.flag_auto_slight_moving = True
                else :
                    if self.count_auto_slight_move > 90 :
                        self.on_click_slightly_circle()
                        self.on_click_reset_cam()
                        self.on_click_pause()
                        self.count_auto_slight_move = 0
                        self.flag_auto_slight_moving = False

            if self.status_azim_circle :
                self.azim_add()
            elif self.status_slightly_circle :
                self.slightly_circle(self.count_sligtly_circle)
                self.count_sligtly_circle = self.count_sligtly_circle + 1
            self.count_frame = self.count_frame + 1
            data_in = queue_out.get()
            img_geo_tensor = data_in['main']['render'] * 255
            img_geo = img_geo_tensor.to(torch.uint8).cpu().numpy().astype(np.uint8)
            img_seg = data_in['vis']['seg']
            img_det = data_in['vis']['det']

            if data_in['flag_save'] and self.flag_save_input:
                self.saving_original_input(data_in['vis']['original'], data_in['vis']['original_mask'])
                self.flag_saving_continous = True
            else :
                self.flag_saving_continous = False

            self.update_all_frame(
                res_geo=img_geo, 
                res_tex=None, 
                vis_det=img_det, 
                vis_seg=img_seg
            )
        
        if self.count_main >= 20 :
            self.update_fps(self.count_frame)
            self.count_main = 0
            self.count_frame = 0

        for i in range(len(self.list_queue)) :
            self.list_qsize[i] = self.list_queue[i].qsize()
        self.update_qsize()
        #self.main_window.update_fps(fps)


        
        


 