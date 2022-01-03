import os
import multiprocessing
import cv2 as cv

class Process_Streamer(multiprocessing.Process) :
    def __init__(self, queue_next, path_source=None, name='', rotate=0, fps_expect=15, streamer_controller=None, source='video') :
        super().__init__()
        self.name = name
        self.queue_next = queue_next
        self.rotate = rotate
        self.fps_expect = fps_expect
        self.path_source = path_source
        self.streamer_controller = streamer_controller
        self.source = source
        self.interval = 1 / self.fps_expect
        self.flag_pause = False

        assert self.source in ['video', 'webcam', 'imgs']
        if self.source in ['video', 'imgs'] :
            assert self.path_source is not None
        return

    def run(self) :
        print('Process ' + self.name +  ' started.')
        if self.source == 'video' :
            streamer = Video_streamer(self.path_source, self.rotate)
        elif self.source == 'webcam' :
            streamer = WebCam_streamer(cam_id=0, rotate=self.rotate)
        elif self.source == 'imgs' :
            streamer = Image_Folder_Streamer(self.path_source, self.rotate)
        event = multiprocessing.Event()

        while(True) :
            if self.streamer_controller is not None :
                if not self.streamer_controller.empty() :
                    self.flag_pause = self.streamer_controller.get()
            if not self.flag_pause :
                data_out = streamer.next()
            self.queue_next.put(data_out)
            if self.source != 'webcam' :
            	event.wait(self.interval)
        return


class Process_Streamer_Video(multiprocessing.Process) :
    def __init__(self, queue_next, path_video, name='', rotate=0, fps_expect=15, streamer_controller=None) :
        super().__init__()
        self.name = name
        self.queue_next = queue_next
        self.rotate = rotate
        self.fps_expect = fps_expect
        self.path_video = path_video
        self.streamer_controller = streamer_controller

        self.interval = 1 / self.fps_expect

        self.flag_pause = False
        return

    def run(self) :
        streamer = Video_streamer(self.path_video, self.rotate)
        #streamer = WebCam_streamer(rotate=self.rotate)
        #streamer = Image_Folder_Streamer(self.path_video, self.rotate)
        event = multiprocessing.Event()

        while(True) :
            if self.streamer_controller is not None :
                if not self.streamer_controller.empty() :
                    self.flag_pause = self.streamer_controller.get()
            if not self.flag_pause :
                data_out = streamer.next()
            self.queue_next.put(data_out)
            event.wait(self.interval)
        return

class Base_streamer() :
    def __init__(self) :
        self.length = None
        return

    def next(self) :
        return None

    def get_len(self) :
        return self.length

class WebCam_streamer(Base_streamer) :
    def __init__(self, rotate, cam_id=0, length=1_0000_0000_0000) :
        self.length = length
        self.cam_id = cam_id
        self.rotate = rotate
        self.cap = cv.VideoCapture(self.cam_id) 
        self.W = 1920
        self.H = 1080
        self.cap.set(3, self.W)
        self.cap.set(4, self.H)

    def next(self) :
        ret, frame_bgr = self.cap.read()
        if not ret:
            print('Can\'t receive frame (stream end?).')
        else :
            if self.rotate == 270 :
                frame_bgr = cv.transpose(frame_bgr)
                frame_bgr = cv.flip(frame_bgr, 1)
                
                #frame_bgr[:, :135,:] = 0
                #frame_bgr[:, -135:,:] = 0
        return frame_bgr

class Video_streamer(Base_streamer) :
    def __init__(self, path_video, rotate) :
        self.cap = cv.VideoCapture(path_video)
        self.path_video = path_video
        self.rotate = rotate
        self.length = int(self.cap.get(7))

    def next(self) :
        ret, frame_bgr = self.cap.read()
        if not ret:
            print('Can\'t receive frame (stream end?).')
            #exit()
            frame_bgr = None
            self.cap.release()
            self.cap = cv.VideoCapture(self.path_video)
            return self.next()
        else :
            if self.rotate == 270 :
                frame_bgr = cv.transpose(frame_bgr)
                frame_bgr = cv.flip(frame_bgr, 1)
        return frame_bgr
    def get_length(self) :
        return self.length

class Image_Folder_Streamer(Base_streamer) :
    def __init__(self, path_folder, rotate) :
        self.path_folder = path_folder
        self.list_dir = os.listdir(self.path_folder)
        self.rotate = rotate
        self.length = len(self.list_dir)
        self.present_index = 0

    def next(self) :
        path_img = os.path.join(self.path_folder, self.list_dir[self.present_index])
        self.present_index = (self.present_index + 1) % self.length
        frame_bgr = cv.imread(path_img)
        if self.rotate == 270 :
            frame_bgr = cv.transpose(frame_bgr)
            frame_bgr = cv.flip(frame_bgr, 1)
        return frame_bgr
    def get_length(self) :
        return self.length


