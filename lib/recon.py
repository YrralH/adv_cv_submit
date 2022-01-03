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


class Process_Recon(multiprocessing.Process) :
    def __init__(self, queue_prev, queue_next, name='', device=torch.device('cuda:1')) :
        super().__init__()
        self.name = name
        self.queue_prev = queue_prev
        self.queue_next = queue_next
        self.device = device
        return

    def run(self) :
        print('Process ' + self.name +  ' started.')
        with torch.no_grad() :
            import lib.opt_recon
            opt = lib.opt_recon.default_opt
            engine = Evaluator(opt, self.device)

            while(True) :
                data_in = self.queue_prev.get()
                data_net = engine.load_tensor(data_in['main']['res'])
                if data_net is not None :
                    sdf = engine.gen_sdf_hj_fast(data_net)
                else :
                    sdf = None

                data_main = {'sdf': sdf}
                data_out = {'main': data_main, 'vis':data_in['vis']}
                self.queue_next.put(data_out)
            return



class Evaluator:
    def __init__(self, opt, device, projection_mode='perspective'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')
        cuda = device

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        #print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        #if opt.load_netC_checkpoint_path is not None:
        #    print('loading for net C ...', opt.load_netC_checkpoint_path)
        #    netC = ResBlkPIFuNet(opt).to(device=cuda)
        #    netC.load_state1_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        #else:
        #    netC = None

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        #self.netC = netC
        self.netG.eval()

        #recon engine using monoport sampling strategy
        def query_func(points, _calib) :
            '''
            :para point [1, N, 3] ([B, C, N])
            :para calib [1, 4, 4] ([B, 4*4matrix])
            :return [1, 1, N] ([B, C, N])
            '''
            samples = points.permute(0, 2, 1) # [bz, 3, N
            self.netG.query(samples, _calib)
            return self.netG.get_preds()
        self.reconEngine = self.get_recon_engine(query_func)
        return


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
        ).to(self.cuda)
        return reconEngine


    def load_tensor(self, image):
        '''
        image: [1,4,1920,1080]
        '''
        #print(image.shape)
        # crop here: crop to [1,4,1480,1480]
        if image.shape[1] == 4:
            image = torch.nn.functional.pad(image[:,:,250:250+1480,:],(200,200),mode='constant', value=0)
            # resize: resize to [1,4,512,512]
            image = torch.nn.functional.interpolate(image, size = (512,512), mode = "bilinear", align_corners=False).to(self.cuda)
            mask = (image[:,3:4,:,:])
            image = (image[:,0:3,:,:]*mask)
        elif image.shape[1] == 3 :
            image = torch.nn.functional.pad(image[:,:,250:250+1480,:],(200,200),mode='constant', value=0)
            image = torch.nn.functional.interpolate(image, size = (512,512), mode = "bilinear", align_corners=False).to(self.cuda)


        calib = torch.Tensor(PROJECTION_MATRIX).to(torch.float32).to(self.cuda)
        image = image.to(self.cuda)
        #mask = mask.to(self.cuda)
        # 注意 calib,image和mask在GPU
        return {
            'name': "",
            'img': image,
            'calib': calib.unsqueeze(0),
            'b_min': np.array([-1, -1, -1]),
            'b_max': np.array([1, 1, 1]),
        }




    def gen_sdf_hj_fast(self, data):
        '''
        created by hj in 2021-04-27
        test fast gen_mesh
        '''
        with torch.no_grad() :
            #prepare data
            RES = self.opt.resolution
            image_tensor = data['img']
            calib_tensor = data['calib']

            #get feature
            #print('image_tensor.shape', image_tensor.shape)
            self.netG.filter(image_tensor)

            #declare sdf tensor 
            sdf = self.reconEngine(_calib=calib_tensor) #[1, 1, 257, 257, 257])
            if sdf is not None :
                return (sdf[0, 0, 0:256, 0:256, 0:256]).contiguous()#[256, 256, 256])
            else :
            	return None





            
