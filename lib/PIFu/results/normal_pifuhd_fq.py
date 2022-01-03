import os
import torch
import numpy as np
from tqdm import tqdm
#import imageio
import cv2
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt
#from skimage import img_as_ubyte
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
# io utils
from pytorch3d.io import load_obj, save_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    PerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, SoftPhongShader, PointLights, TexturesVertex,
)

from ColorShader import ColorShader, NormalShader
import cv2
# Set the cuda device 
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

R, T = look_at_view_transform(dist=2, elev=0, azim=0, degrees=True)
fx = 2.1853125*256
fy = 2.1853125*256
px = 256
py = 256
cameras = PerspectiveCameras(
    device=device, 
    focal_length=((fx, fy),),
    principal_point=((512-px, 512-py),),
    image_size=((512, 512),),
    R=R.to(device), 
    T=T.to(device)
)

blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1, 1, 1)) 
raster_settings = RasterizationSettings(
    image_size=(512,512), 
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma / 10, 
    faces_per_pixel=100, 
)
normal_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=NormalShader(blend_params=blend_params, cameras=cameras)
)

for i in range(663):
    verts, faces_idx, _ = load_obj("pifu_demo/result_%04d.obj"%(i,))
    faces = faces_idx.verts_idx
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    verts_rgb[0, :, 1:] = 0
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    pred_mesh = Meshes(
        verts=[verts.to(device)],   
        faces=[faces.to(device)],
        textures=textures
    )

    tmp = torch.clip((normal_renderer(pred_mesh, cameras=cameras)[0,:,:, :3]*0.5+0.5)*255,0,255).cpu().numpy()
    tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB) 
    cv2.imwrite("norml/%04d.png"%(i,),tmp)
    print(i)

