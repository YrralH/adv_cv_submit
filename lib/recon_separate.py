import sys
import os

import multiprocessing
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.algorithm.imageProcess import hj_resize_to

from lib.PIFu.lib.options import BaseOptions
from lib.PIFu.lib.mesh_util import *
from lib.PIFu.lib.sample_util import *
from lib.PIFu.lib.train_util import *
from lib.PIFu.lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

from lib.ImplicitSegCUDA.implicit_seg.functional import Seg3dLossless


PROJECTION_MATRIX = np.array([  [2.1853125,0,0,0],
                                [0,-2.1853125,0,0],
                                [0,0,-1,2.5],
                                [0,0,0,0]])

class Process_Recon_Double(multiprocessing.Process) :
    def __init__(self, queue_prev, queue_next, name='') :
        super().__init__()
        self.name = name
        self.queue_prev = queue_prev
        self.queue_next = queue_next
        return

    def run(self) :
        cuda0 = torch.device('cuda:0')
        cuda1 = torch.device('cuda:1')

        engine_filter_0 = Image_Filter_Engine(cuda0)
        engine_filter_1 = Image_Filter_Engine(cuda1)
        engine_Gen_SDF_0 = Gen_SDF_Engine(cuda0)
        engine_Gen_SDF_1 = Gen_SDF_Engine(cuda1)

        while(True) :
            data_in_0 = self.queue_prev.get()
            data_in_1 = self.queue_prev.get()

            data_img_0 = data_in_0['main']['res']
            data_img_1 = data_in_0['main']['res']
            data_feature_0 = engine_filter_0.process(data_img_0)
            data_feature_1 = engine_filter_1.process(data_img_1)

            data_sdf_0 = engine_Gen_SDF_0.process(data_feature_0)
            data_sdf_1 = engine_Gen_SDF_1.process(data_feature_1)

            data_main_0 = {'sdf': data_sdf_0}
            data_main_1 = {'sdf': data_sdf_1}
            data_out_0 = {'main': data_main_0, 'vis':data_in_0['vis']}
            data_out_1 = {'main': data_main_1, 'vis':data_in_1['vis']}

            self.queue_next.put(data_out_0)
            self.queue_next.put(data_out_1)
        return

class Process_Gen_SDF(multiprocessing.Process) :
    def __init__(self, queue_prev, queue_next, name='', device=torch.device('cuda:1')) :
        super().__init__()
        self.name = name
        self.queue_prev = queue_prev
        self.queue_next = queue_next
        self.device = device
        return

    def run(self) :
        engine = Gen_SDF_Engine(self.device)
        while(True) :
            data_in = self.queue_prev.get()
            data_feature = data_in['main']['feature']
            data_sdf = engine.process(data_feature)

            data_main = {'sdf': data_sdf}
            data_out = {'main': data_main, 'vis':data_in['vis']}
            self.queue_next.put(data_out)
        return

class Process_Image_Filter(multiprocessing.Process) :
    def __init__(self, queue_prev, queue_next, name='', device=torch.device('cuda:1')) :
        super().__init__()
        self.name = name
        self.queue_prev = queue_prev
        self.queue_next = queue_next
        self.device = device
        return

    def run(self) :
        engine = Image_Filter_Engine(self.device)
        while(True) :
            data_in = self.queue_prev.get()
            data_img = data_in['main']['res']
            data_feature = engine.process(data_img)

            data_main = {'feature': data_feature}
            data_out = {'main': data_main, 'vis':data_in['vis']}
            self.queue_next.put(data_out)
        return

class Image_Filter_Engine() :
    def __init__(self, device) :
        path_base = '/home/hj/__workspace__/real_time_multi_process/lib/PIFu/'
        path_relative = 'checkpoints/filter.pkl'
        
        self.path_model = os.path.join(path_base, path_relative)
        #print('self.path_model', self.path_model)
        self.device = device
        self.model = torch.load(self.path_model).to(self.device)
        self.model.eval()

    def process(self, images) :
        image_tmp = torch.nn.functional.pad(images[:,:,320+90:320+90+1280,:],(100,100),mode='constant', value=0)
        image_tmp = torch.nn.functional.interpolate(image_tmp, size = (512,512), mode = "bilinear", align_corners=False).to(self.device)
        image_tmp = (image_tmp[:,0:3,:,:]*image_tmp[:,3:4,:,:])
        image_tmp = image_tmp.to(self.device)
        '''
        calib = torch.Tensor(PROJECTION_MATRIX).to(torch.float32).to(self.cuda)
        '''
        feature_list ,____ ,____ = self.model(image_tmp)
        return feature_list[-1]

class Gen_SDF_Engine() :
    def __init__(self, device) :
        path_base = '/home/hj/__workspace__/real_time_multi_process/lib/PIFu/'
        path_relative = 'checkpoints/mlp.pkl'
        sys.path.insert(0, path_base)

        self.path_model = os.path.join(path_base, path_relative)
        #print('self.path_model', self.path_model)
        self.device = device
        self.model = torch.load(self.path_model).to(self.device)
        self.calib = self.get_calib()

        self.model.eval()

        def query_function(_model, points, _calib, _im_feat):
            '''
            :para points [1, N, 3] ([B, C, N])
            :para _calib [1, 4, 4] ([B, 4*4matrix])
            :para _im_feat [1, 256, 128, 128]
            :return [1, 1, N] ([B, C, N])
            '''
            #print('points', points.shape)
            #print('_calib', _calib.shape)
            #print('_im_feat', _im_feat.shape)

            #additonal
            points = points.permute(0, 2, 1)

            rot = _calib[:, :3, :3]
            trans = _calib[:, :3, 3:4]
            homo = torch.baddbmm(trans, rot, points)
            xy = homo[:, :2, :] / homo[:, 2:3, :]
            z = homo[:, 2:3, :] - trans[:, 2:3, :]

            in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
            # 注意：以lib/model/DepthNormalizer.py里的实现为准，后续版本这里变了是 z_feat = z
            z_feat = z * (256 / 200 *1.28) 
            samples = torch.nn.functional.grid_sample(
                _im_feat, 
                xy.transpose(1, 2).unsqueeze(2), 
                # 注意：以你lib/geometry.py里index函数的为准，后续版本我都是False
                align_corners=True
            ).squeeze(3)
            point_local_feat = torch.cat([samples, z_feat], 1)
            pred = in_img[:,None].float() * _model(point_local_feat)
            return pred
        self.recon_Engine = self.get_recon_engine(query_function)

    def get_calib(self) :
        calib = torch.Tensor(PROJECTION_MATRIX).to(torch.float32).to(self.device)
        return calib[None] #[1, 4, 4]

    def process(self, feature) :
        '''
        :para feature
        :return sdf
        '''
        sdf = self.recon_Engine(_model=self.model, _calib=self.calib, _im_feat=feature) #[1, 1, 257, 257, 257])
        return (sdf[0, 0, 0:256, 0:256, 0:256]).contiguous()#[256, 256, 256])

    def get_recon_engine(self, _query_func) :
        b_min_tensor = torch.tensor([-1.0, -1.0, -1.0]).float()
        b_max_tensor = torch.tensor([ 1.0,  1.0,  1.0]).float()
        resolutions = [16+1, 32+1, 64+1, 128+1, 256+1]
        reconEngine = Seg3dLossless(
            query_func=_query_func, 
            b_min=b_min_tensor.unsqueeze(0).numpy(),
            b_max=b_max_tensor.unsqueeze(0).numpy(),
            resolutions=resolutions,
            balance_value=0.5,
            use_cuda_impl=False,
            faster=True
        ).to(self.device)
        return reconEngine
