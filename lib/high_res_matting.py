
import multiprocessing

import numpy as np
import cv2 as cv
import torch

from lib.BackgroundMattingV2.model import MattingBase, MattingRefine

class Process_Pre_Process2(multiprocessing.Process) :
    def __init__(self, queue_prev, queue_next, path_bg, name='', device=torch.device('cuda:0')) :
        super().__init__()
        self.name = name
        self.queue_prev = queue_prev
        self.queue_next = queue_next
        self.path_bg = path_bg
        self.device = device
        return

    def run(self) :
        print('Process ' + self.name +  ' started.')
        with torch.no_grad() :
            engine = Matting_Engine(self.device, self.path_bg)

            while(True) :
                data_in = self.queue_prev.get()
                data_out = engine.process(data_in)
                self.queue_next.put(data_out)
            return

class Matting_Engine() :
    def __init__(self, device, path_bg) :
        PATH_CKP_RES50 = '/home/hj/__workspace__/real_time_multi_process/lib/BackgroundMattingV2/checkpoint/pytorch_resnet50.pth'
        self.device = device
        self.model = MattingRefine(
            backbone='resnet50',
            backbone_scale=0.25,
            refine_mode='sampling',
            refine_sample_pixels=80000,
            refine_kernel_size=3
        ).to(device).eval()
        self.model.load_state_dict(
            torch.load(
                PATH_CKP_RES50, 
                map_location=self.device
            ), 
            strict=False
        )

        if type(path_bg) == str :
            bg_bgr = cv.imread(path_bg)
        else :
            bg_bgr = path_bg
        bg_rgb = cv.cvtColor(bg_bgr, cv.COLOR_BGR2RGB) 
        bg_rgb_torch = torch.from_numpy(bg_rgb).to(self.device).permute(2, 0, 1)[None] #[1, 3, 1920, 1080]
        bg_rgb_torch = bg_rgb_torch / 255.0 #(0 - 1)
        self.bg = bg_rgb_torch
        self.count = 0

    def process(self, data_in) :
        original = data_in
        #print('matting')
        #print(data_in.shape)
        #print(self.bg.shape)
        frame_rgb = cv.cvtColor(original, cv.COLOR_BGR2RGB) 
        frame_rgb_torch = torch.from_numpy(frame_rgb).to(self.device).permute(2, 0, 1)[None] #[1, 3, 1920, 1080]
        frame_rgb_torch = frame_rgb_torch / 255.0 #(0 - 1)
        
        pha, fgr, ____, ____, ____, ____ = self.model(frame_rgb_torch, self.bg)

        #vis_seg = torch.nn.functional.pad(pha[:,:,250:250+1480,:],(200,200),mode='constant', value=0)
        pha_cut = pha[:,:,250:250+1480,:]
        vis_seg = torch.nn.functional.interpolate(pha_cut, size = (411,300), mode = "bilinear", align_corners=False)
        vis_seg = vis_seg[0] #(1, 448, 378)
        vis_seg = vis_seg.permute(1, 2, 0).contiguous() #(448, 378, 1)
        vis_seg = (vis_seg * 255).to(torch.uint8).cpu().numpy().astype(np.uint8) #(448, 378, 1)

        #vis_det = torch.nn.functional.pad(fgr[:,:,250:250+1480,:],(200,200),mode='constant', value=0)
        frame_rgb_torch_cut = frame_rgb_torch[:,:,250:250+1480,:]
        vis_det = torch.nn.functional.interpolate(frame_rgb_torch_cut, size = (411,300), mode = "bilinear", align_corners=False)
        vis_det = vis_det[0] #(3, 448, 378)
        vis_det = vis_det.permute(1, 2, 0).contiguous() #(448, 378, 3)
        vis_det = (vis_det * 255).to(torch.uint8).cpu().numpy().astype(np.uint8) #(448, 378, 3)

        fgr = fgr * 2.0 - 1.0
        res = fgr * pha

        data_main = {'res': res}
        data_vis = {
            'seg': vis_seg, 
            'det': vis_det, 
            'original': frame_rgb_torch_cut[0].cpu(),
            'original_mask':pha_cut[0].cpu(),
            'count':self.count
        }
        self.count = self.count + 1
        data = {'main': data_main, 'vis': data_vis}
        return data



