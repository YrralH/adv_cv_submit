import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

# get options

opt = BaseOptions().parse()
print(type(opt))
with open('/home/hj/__workspace__/real_time_front_end/lib/PIFu/hj_opt.pkl', 'wb') as f:  
    pickle.dump(opt, f)
with open('/home/hj/__workspace__/real_time_front_end/lib/PIFu/hj_opt.pkl', 'rb') as f:  
    opt = pickle.load(f)
print(type(opt))
print(opt)
exit()

class Evaluator:
    def __init__(self, opt, projection_mode='perspective'):
        self.opt = opt
        self.load_size = self.opt.loadSize
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # set cuda
        cuda = torch.device('cuda:%d' % opt.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

        # create net
        netG = HGPIFuNet(opt, projection_mode).to(device=cuda)
        print('Using Network: ', netG.name)

        if opt.load_netG_checkpoint_path:
            netG.load_state_dict(torch.load(opt.load_netG_checkpoint_path, map_location=cuda))

        if opt.load_netC_checkpoint_path is not None:
            print('loading for net C ...', opt.load_netC_checkpoint_path)
            netC = ResBlkPIFuNet(opt).to(device=cuda)
            netC.load_state_dict(torch.load(opt.load_netC_checkpoint_path, map_location=cuda))
        else:
            netC = None

        os.makedirs(opt.results_path, exist_ok=True)
        os.makedirs('%s/%s' % (opt.results_path, opt.name), exist_ok=True)

        opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
        with open(opt_log, 'w') as outfile:
            outfile.write(json.dumps(vars(opt), indent=2))

        self.cuda = cuda
        self.netG = netG
        self.netC = netC

    def load_tensor(self, image):
        '''
        image: [1,4,1920,1080]
        '''
        # crop here: crop to [1,4,1280,1280]
        image = torch.nn.functional.pad(image[:,320+90:320+90+1280,:,:],(100,100),mode='constant', value=0)
        # resize: resize to [1,4,512,512]
        image = torch.nn.functional.interpolate(image, size = (512,512), mode = "bilinear", align_corners=False)
        mask = image[:,3:4,:,:]
        image= image[:,0:3,:,:]*mask
        projection_matrix = np.array([  [2.1853125,0,0,0],
                                        [0,-2.1853125,9/64,-9/64],
                                        [0,0,-1,2],
                                        [0,0,0,0]])
        calib = torch.Tensor(projection_matrix).float()
        # ?????? calib??????GPU??????image???mask???
        return {
            'name': "",
            'img': image,
            'calib': calib.unsqueeze(0),
            'mask': mask,
            'b_min': np.array([-1, -1, -1]),
            'b_max': np.array([1, 1, 1]),
        }

    def load_image(self, image_path, mask_path):
        # Name
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        # Calib
        B_MIN = np.array([-1, -1, -1])
        B_MAX = np.array([1, 1, 1])
        projection_matrix = np.array([  [2.1853125,0,0,0],
                                        [0,-2.1853125,0,0],
                                        [0,0,-1,2],
                                        [0,0,0,0]])
        calib = torch.Tensor(projection_matrix).float()
        # Mask
        mask = Image.open(mask_path).convert('L')
        mask = transforms.Resize(self.load_size)(mask)
        mask = transforms.ToTensor()(mask).float()
        # image
        image = Image.open(image_path).convert('RGB')
        image = self.to_tensor(image)
        image = mask.expand_as(image) * image
        return {
            'name': img_name,
            'img': image.unsqueeze(0),
            'calib': calib.unsqueeze(0),
            'mask': mask.unsqueeze(0),
            'b_min': B_MIN,
            'b_max': B_MAX,
        }

    def eval(self, data, use_octree=False):
        '''
        Evaluate a data point
        :param data: a dict containing at least ['name'], ['image'], ['calib'], ['b_min'] and ['b_max'] tensors.
        :return:
        '''
        opt = self.opt
        with torch.no_grad():
            self.netG.eval()
            if self.netC:
                self.netC.eval()
            save_path = '%s/%s/result_%s.obj' % (opt.results_path, opt.name, data['name'])
            print('gen mesh')
            gen_mesh(opt, self.netG, self.cuda, data, save_path, use_octree=use_octree)
            #if self.netC:
            #    gen_mesh_color(opt, self.netG, self.netC, self.cuda, data, save_path, use_octree=use_octree)
            #else:
            #   gen_mesh(opt, self.netG, self.cuda, data, save_path, use_octree=use_octree)


if __name__ == '__main__':
    evaluator = Evaluator(opt)

    test_images = glob.glob(os.path.join(opt.test_folder_path, '*'))
    test_images = [f for f in test_images if ('png' in f or 'jpg' in f) and (not 'mask' in f)]
    test_masks = [f[:-4]+'_mask.png' for f in test_images]

    print("num; ", len(test_masks))

    for image_path, mask_path in tqdm.tqdm(zip(test_images, test_masks)):
        try:
            print(image_path, mask_path)
            data = evaluator.load_image(image_path, mask_path)
            evaluator.eval(data, True)
        except Exception as e:
           print("error:", e.args)
        exit()
