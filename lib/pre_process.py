
import multiprocessing

import numpy as np
import cv2 as cv
import torch

from lib.algorithm.imageProcess import hj_resize_to

import human_inst_seg

class Process_Pre_Process(multiprocessing.Process) :
    def __init__(self, queue_prev, queue_next, name='', device=torch.device('cuda:0')) :
        super().__init__()
        self.name = name
        self.queue_prev = queue_prev
        self.queue_next = queue_next
        self.device = device
        return

    def run(self) :
        print('Process ' + self.name +  ' started.')
        with torch.no_grad() :
            engine = Pre_Process_Engine(self.device)

            while(True) :
                data_in = self.queue_prev.get()
                data_out = engine.process(data_in)
                self.queue_next.put(data_out)
            return

class Pre_Process_Engine() :
    def __init__(self, device, name='') :
        self.device = device
        self.seg_engine = human_inst_seg.Segmentation()
        self.seg_engine.eval()
        self.count = 0
        return
    
    def process(self, data_in) :
        #print(type(data_in)) #<class 'numpy.ndarray'>
        #original = cv.UMat(data_in)
        original = data_in
        frame_rgb = cv.cvtColor(original, cv.COLOR_BGR2RGB) 
        frame_rgb_torch = torch.from_numpy(frame_rgb).to(self.device).permute(2, 0, 1)[None] #[1, 3, 1920, 1080]
        frame_rgb_torch = frame_rgb_torch / 127.5 - 1
        
        outputs, bboxes, probs = self.seg_engine(frame_rgb_torch)
        #print('seg_engine, outputs.shape', outputs.shape) #torch.Size([1, 4, 1920, 1080])
        #print('seg_engine, bboxes.shape', bboxes.shape) #torch.Size([1, 1, 1, 4])
        #print('seg_engine, probs.shape', probs.shape) #torch.Size([1, 1, 1])
        bbox = bboxes[0, 0, 0]
        res = outputs

        def crop_to_bbox() :
            #check valid
            MAX_H = 1024
            MAX_W = 1024
            flag_prob_low = (probs[0,0,0] < 0.8).cpu().numpy()
            flag_bbox_err = 0 > int(bbox[0]) or 0 > int(bbox[1]) or \
                            0 > int(bbox[2]) or 0 > int(bbox[3])
            h = int(bbox[3]) - int(bbox[1])
            w = int(bbox[2]) - int(bbox[0])
            size = h if h > w else w
            size = size * 1.15
            h1 = max(int(bbox[1]) - (size - h) // 2, 0)
            h2 = min(int(bbox[3]) + (size - h) // 2, MAX_H)
            w1 = max(int(bbox[0]) - (size - w) // 2, 0)
            w2 = min(int(bbox[2]) + (size - w) // 2, MAX_W)

            flag_too_small = h2 - h1 < 30 or w2 - w1 < 15
            print(flag_prob_low, flag_bbox_err, flag_too_small)
            if  flag_prob_low or flag_bbox_err or flag_too_small :
                #res = res[:, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                #print('res', torch.max(res), torch.min(res))
                res = None
            else :
                res = res[:, int(h1):int(h2), int(w1):int(w2)]
            return
        #crop_to_bbox
        

        data_vis = None
        #-----------get vis begin-----------
        vis = outputs[:,:,250:250+1480,:]
        vis = torch.nn.functional.interpolate(vis, size = (411,300), mode = "bilinear", align_corners=False)
        vis = vis[0] #(4, 448, 378)

        vis_seg = vis[3:4].permute(1, 2, 0).contiguous()
        vis_seg = (vis_seg * 255).to(torch.uint8).cpu().numpy().astype(np.uint8)

        vis_det = vis[0:3, :, :]
        vis_det = ((vis_det.permute(1, 2, 0).contiguous() + 1) * 127.5).to(torch.uint8).cpu().numpy().astype(np.uint8) 
    
        '''
        bbox[1] = bbox[1] - (320+90) 
        bbox[3] = bbox[3] - (320+90) 
        bbox = bbox / 1280.0 * 448.0
        vis_det = cv.rectangle(
            cv.UMat(vis_det[:,:,0:3]), 
            (int(bbox[0]), int(bbox[1])), 
            (int(bbox[2]), int(bbox[3])), 
            (255,0,0), 
            2
        ).get()
        '''

        vis_probs = probs[0, 0, 0].cpu().numpy()
        #print('vis_seg', vis_seg.shape)
        #print('vis_det', vis_det.shape)
        #print('vis_probs', vis_probs)
        #print('get_vis, vis.shape', vis.shape)
        #-----------get vis end-----------   

        data_main = {'res': res}
        data_vis = {'seg': vis_seg, 'det': vis_det, 'count':self.count}
        self.count = self.count + 1
        data = {'main': data_main, 'vis': data_vis}

        #print('data_main', res.shape)

        return data

    def process_batch(self, list_data_in, batch_size) :
        list_frame_rgb_torch = []
        for i in range(batch_size) :
            frame_rgb = cv.cvtColor(list_data_in[i], cv.COLOR_BGR2RGB).astype(np.float32)
            frame_rgb_torch = torch.tensor(frame_rgb, device=self.device).permute(2, 0, 1) / 127.5 - 1
            list_frame_rgb_torch.append(frame_rgb_torch)
        batch_input_frame_rgb_torch = torch.stack(list_frame_rgb_torch, dim=0)

        outputs, bboxes, probs = self.seg_engine(batch_input_frame_rgb_torch)
        
        list_data = []
        for i in range(batch_size) :
            data_main = {'res': outputs[i][None]}
            data_vis = None
            data = {'main': data_main, 'vis': data_vis}
            list_data.append(list_data)
        return list_data


