from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging

log = logging.getLogger('trimesh')
log.setLevel(40)

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


class FQTrainDataset(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt, phase='train'):
        self.opt = opt
        self.projection_mode = 'perspective'
        self.is_train = (phase == 'train')

        # Path setup
        self.root = self.opt.dataroot
        '''
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.UV_MASK = os.path.join(self.root, 'UV_MASK')
        self.UV_NORMAL = os.path.join(self.root, 'UV_NORMAL')
        self.UV_RENDER = os.path.join(self.root, 'UV_RENDER')
        self.UV_POS = os.path.join(self.root, 'UV_POS')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')
        '''
        self.B_MIN = np.array([-1, -1, -1])
        self.B_MAX = np.array([1, 1, 1])

        self.load_size = self.opt.loadSize  # default: 512
        self.num_views = self.opt.num_views # default: 1
        
        self.num_sample_inout = self.opt.num_sample_inout # default: 5000
        self.num_sample_color = self.opt.num_sample_color # default: 0

        self.yaw_list = list(range(36))
        self.pitch_list = [0]
        self.subjects = np.genfromtxt(os.path.join(self.root, phase+".txt"), dtype=np.str).reshape((-1,))

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        '''
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])
        '''


    def __len__(self):
        return len(self.subjects) * len(self.yaw_list) * len(self.pitch_list)

    def get_render(self, subject, num_views, yid=0, pid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''
        pitch = self.pitch_list[pid]
        # all views have the same pitch, we do not use pitch here
        # The ids are an even distribution of num_views around view_id
        view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
                    for offset in range(num_views)]
        if random_sample:
            view_ids = np.random.choice(self.yaw_list, num_views, replace=False)

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []

        for vid in view_ids:
            color_index = np.random.randint(4)
            mask_path = os.path.join(self.root,"mask",subject,"%03d_%03d_%03d.png"%(vid*10,0,color_index))
            render_path = os.path.join(self.root,"image",subject,"%03d_%03d_%03d.png"%(vid*10,0,color_index))
            intrinsic_path = os.path.join(self.root,"intrinsic",subject,"%03d_%03d_%03d.npy"%(vid*10,0,color_index))
            extrinsic_path = os.path.join(self.root,"extrinsic",subject,"%03d_%03d_%03d.npy"%(vid*10,0,color_index))

            mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')
            intrinsic = np.load(intrinsic_path)
            extrinsic = np.load(extrinsic_path)

            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()
            mask_list.append(mask)

            render = self.to_tensor(render)
            render = mask.expand_as(render) * render
            render_list.append(render)

            calib = np.matmul(intrinsic, extrinsic)
            calib[0:2] *= -1
            calib = torch.Tensor(calib).float()
            calib_list.append(calib)

            extrinsic = torch.Tensor(extrinsic).float()
            extrinsic_list.append(extrinsic)
            # you can do some augmentation here
    
        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'mask': torch.stack(mask_list, dim=0)
        }

    # get_inout_sampling is a better name
    def select_sampling_method(self, subject):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)
        mesh_path = os.path.join(self.root,"raw",subject,"mesh.npz")
        mesh = np.load(mesh_path)
        mesh = trimesh.Trimesh(vertices=mesh["v"], faces=mesh["f"], process = False)
        surface_points, _ = trimesh.sample.sample_surface(mesh, 4 * self.num_sample_inout)
        sample_points = surface_points + np.random.normal(scale=self.opt.sigma, size=surface_points.shape)

        # add random points within image space
        length = self.B_MAX - self.B_MIN
        random_points = np.random.rand(self.num_sample_inout // 4, 3) * length + self.B_MIN
        sample_points = np.concatenate([sample_points, random_points], 0)
        np.random.shuffle(sample_points)

        inside = mesh.contains(sample_points)
        inside_points = sample_points[inside]
        outside_points = sample_points[np.logical_not(inside)]

        nin = inside_points.shape[0]
        inside_points = inside_points[
                        :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points
        outside_points = outside_points[
                         :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points[
                                                                                               :(self.num_sample_inout - nin)]

        samples = np.concatenate([inside_points, outside_points], 0).T
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))], 1)

        # save_samples_truncted_prob('out.ply', samples.T, labels.T)
        # exit()

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()
        
        del mesh
        return {
            'samples': samples,
            'labels': labels
        }

    '''
    def get_color_sampling(self, subject, yid, pid=0):
        yaw = self.yaw_list[yid]
        pitch = self.pitch_list[pid]
        uv_render_path = os.path.join(self.UV_RENDER, subject, '%d_%d_%02d.jpg' % (yaw, pitch, 0))
        uv_mask_path = os.path.join(self.UV_MASK, subject, '%02d.png' % (0))
        uv_pos_path = os.path.join(self.UV_POS, subject, '%02d.exr' % (0))
        uv_normal_path = os.path.join(self.UV_NORMAL, subject, '%02d.png' % (0))

        # Segmentation mask for the uv render.
        # [H, W] bool
        uv_mask = cv2.imread(uv_mask_path)
        uv_mask = uv_mask[:, :, 0] != 0
        # UV render. each pixel is the color of the point.
        # [H, W, 3] 0 ~ 1 float
        uv_render = cv2.imread(uv_render_path)
        uv_render = cv2.cvtColor(uv_render, cv2.COLOR_BGR2RGB) / 255.0

        # Normal render. each pixel is the surface normal of the point.
        # [H, W, 3] -1 ~ 1 float
        uv_normal = cv2.imread(uv_normal_path)
        uv_normal = cv2.cvtColor(uv_normal, cv2.COLOR_BGR2RGB) / 255.0
        uv_normal = 2.0 * uv_normal - 1.0
        # Position render. each pixel is the xyz coordinates of the point
        uv_pos = cv2.imread(uv_pos_path, 2 | 4)[:, :, ::-1]

        ### In these few lines we flattern the masks, positions, and normals
        uv_mask = uv_mask.reshape((-1))
        uv_pos = uv_pos.reshape((-1, 3))
        uv_render = uv_render.reshape((-1, 3))
        uv_normal = uv_normal.reshape((-1, 3))

        surface_points = uv_pos[uv_mask]
        surface_colors = uv_render[uv_mask]
        surface_normal = uv_normal[uv_mask]

        if self.num_sample_color:
            sample_list = random.sample(range(0, surface_points.shape[0] - 1), self.num_sample_color)
            surface_points = surface_points[sample_list].T
            surface_colors = surface_colors[sample_list].T
            surface_normal = surface_normal[sample_list].T

        # Samples are around the true surface with an offset
        normal = torch.Tensor(surface_normal).float()
        samples = torch.Tensor(surface_points).float() \
                  + torch.normal(mean=torch.zeros((1, normal.size(1))), std=self.opt.sigma).expand_as(normal) * normal

        # Normalized to [-1, 1]
        rgbs_color = 2.0 * torch.Tensor(surface_colors).float() - 1.0

        return {
            'color_samples': samples,
            'rgbs': rgbs_color
        }
    '''
    def get_item(self, index):
        sid = index % len(self.subjects)   # subject
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw_list)     # yaw
        pid = tmp // len(self.yaw_list)    # pitch

        subject = self.subjects[sid]       # filename
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.root, "raw", subject,'mesh.npz'),
            'sid': sid,
            'yid': yid,
            'pid': pid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }

        '''
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''
        render_data = self.get_render(subject, num_views=self.num_views, yid=yid, pid=pid,
                                        random_sample=self.opt.random_multiview)
        res.update(render_data)

        '''
            'samples': samples
            'labels': labels
        '''
        if self.opt.num_sample_inout: # num of sampling points (for in/out)
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)


        # I don't deal with this now
        '''
            'color_samples': samples
            'rgbs': rgbs_color
        '''
        '''
        if self.num_sample_color: # num of sampling points (for color)
            color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
            res.update(color_data)
        '''
        return res

    def __getitem__(self, index):
        return self.get_item(index)

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from options import BaseOptions
    opt = BaseOptions().parse()
    train_dataset = FQTrainDataset(opt, phase="train")
    for D in train_dataset:
        print(D["calib"])
        break