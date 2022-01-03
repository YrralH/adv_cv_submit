
import sys
import os
import time
import multiprocessing

import cv2 as cv
import numpy as np
import torch

from PyQt5.QtWidgets import QApplication

from lib.streamer import Process_Streamer_Video, Process_Streamer
from lib.pre_process import Process_Pre_Process
from lib.high_res_matting import Process_Pre_Process2
from lib.recon import Process_Recon
from lib.recon_separate import Process_Gen_SDF, Process_Image_Filter, Process_Recon_Double
from lib.sdf2mesh import Process_MCube, Process_MCube_and_Render
from lib.clean_mesh import Process_Clean_Mesh
from lib.render import Process_Render
from lib.main_window import My_MainWindow


def pipeline() :
    multiprocessing.set_start_method('fork')
    app = QApplication(sys.argv)
    data1 = multiprocessing.Queue(2) #frame
    data2 = multiprocessing.Queue(2) #seg res
    data3  = multiprocessing.Queue(1) #sdf
    data4 = multiprocessing.Queue(2) #v, f
    data5 = multiprocessing.Queue(2) #v, f
    data6 = multiprocessing.Queue(2)

    render_controller = multiprocessing.Queue(3)
    streamer_controller = multiprocessing.Queue(3)

    cuda0 = torch.device('cuda:0')
    cuda1 = torch.device('cuda:1')

    list_queue = [data1, data2, data3, data4, data5, data6]


    path_video = '/mnt/data/hj/c920pro/video/103.mp4'
    rotate = 270
    cap = cv.VideoCapture(path_video)
    ret, frame_bgr = cap.read()
    assert ret
    if rotate == 270 :
        frame_bgr = cv.transpose(frame_bgr)
        frame_bgr = cv.flip(frame_bgr, 1)
    bg = frame_bgr
    PATH_BG_DEFAULT = './data/bg/0.png'
    cap.release()

    type_streamer = 'video'
    if type_streamer == 'video' :
        bg_matting = bg
    elif type_streamer == 'webcam' :
        bg_matting = PATH_BG_DEFAULT

    '''
    Process1 = Process_Streamer_Video(
        queue_next=data1, 
        path_video=path_video, 
        name='streamer', 
        rotate=270, 
        fps_expect=20,
        streamer_controller=streamer_controller
    )
    '''

    Process1 = Process_Streamer(
        queue_next=data1, 
        name='streamer', 
        path_source=path_video, 
        rotate=270, 
        fps_expect=30,
        streamer_controller=streamer_controller, 
        source=type_streamer
    )


    '''
    Process2 = Process_Pre_Process(
        queue_prev=data1, 
        queue_next=data2, 
        name='Det and Seg',
        device=cuda0
    )
    '''
    Process2 = Process_Pre_Process2(
        queue_prev=data1, 
        queue_next=data2, 
        name='Seg',
        path_bg=bg_matting,
        device=cuda0
    )


    Process3_0 = Process_Recon(
        queue_prev=data2, 
        queue_next=data3, 
        name='Recon',
        device=cuda0
    )

    '''
    Process3_1 = Process_Recon(
        queue_prev=data2, 
        queue_next=data3, 
        name='Recon',
        device=cuda1
    )
    '''

    Process4_0 = Process_MCube(
        queue_prev=data3, 
        queue_next=data4, 
        name='MCube',
        device=None
    )

    '''
    Process4_1 = Process_MCube(
        queue_prev=data3, 
        queue_next=data4, 
        name='MCube',
        device=None
    )
    '''
    



    Process4_half_00 = Process_Clean_Mesh(queue_prev=data4, queue_next=data5, name='clean mesh')
    #Process4_half_01 = Process_Clean_Mesh(queue_prev=data4, queue_next=data5, name='clean mesh')

    Process5 = Process_Render(
        queue_prev=data5, 
        queue_next=data6, 
        name='render',
        device=cuda0,
        render_controller=render_controller
    )
   
    
    Process1.daemon = True
    Process2.daemon = True
    Process3_0.daemon = True
    #Process3_1.daemon = True
    Process4_0.daemon = True
    #Process4_1.deamon = True
    Process4_half_00.daemon = True
    #Process4_half_01.daemon = True
    Process5.daemon = True
    
    list_forbidden_city = [
        Process1, 
        Process2, 
        Process3_0,
        #Process3_1,
        Process4_0,
        #Process4_1,
        Process4_half_00,
        #Process4_half_01,
        Process5
    ]

    start_all(list_forbidden_city)


    main_window = My_MainWindow(
        list_queue=list_queue, 
        list_forbidden_city=list_forbidden_city, 
        render_controller=render_controller,
        streamer_controller=streamer_controller,
    )

    sys.exit(app.exec_())


def start_all(list_process) :
    for i in list_process :
        i.start()
   
    

def sequence() :
    from lib.streamer import Video_streamer
    from lib.pre_process import Pre_Process_Engine
    from lib.recon import Image_Filter_Engine, Gen_SDF_Engine

    cuda0 = torch.device('cuda:0')
    cuda1 = torch.device('cuda:1')

    stage1 = Video_streamer(path_video='/mnt/data/hj/c920pro/video/4.mp4', rotate=270)
    stage2 = Pre_Process_Engine(device=cuda1)
    stage3 = Image_Filter_Engine(device=cuda1)
    stage4 = Gen_SDF_Engine(device=cuda1)

    tmp1 = stage1.next()
    tmp2 = stage2.process(tmp1)
    tmp3 = stage3.process(tmp2['main']['res'])
    tmp4 = stage4.process(tmp3)
    
    print(tmp4.shape)

def debug_window() :
    app = QApplication(sys.argv)
    data1 = multiprocessing.Queue(4) #frame
    data2 = multiprocessing.Queue(4) #seg res
    data3 = multiprocessing.Queue(1) #sdf
    data5 = multiprocessing.Queue(4) #v, f
    data6 = multiprocessing.Queue(10)
    list_queue = [data1, data2, data3, data5, data6]
    main_window = My_MainWindow(list_queue=list_queue)
    sys.exit(app.exec_())


if __name__ == '__main__' :
    with torch.no_grad() :
        pipeline()


        
