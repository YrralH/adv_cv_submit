import os
import sys
import time
import math
import multiprocessing

#extenal lib
import numpy as np
import cv2 as cv

#pytorch
import torch
import torchvision

#pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras, 
    PointLights, 
    DirectionalLights,
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex
)
from lib.shader import NormalShader


        


class Process_Render(multiprocessing.Process) :
    def __init__(self, queue_prev, queue_next, name='', device=torch.device('cuda:1'), render_controller=None) :
        super().__init__()
        self.name = name
        self.device = device
        self.queue_prev = queue_prev
        self.queue_next = queue_next
        self.render_controller = render_controller
        self.flag_normal_shader = False
        self.flag_pause = False
        self.image_size = 650
        return


    def run(self) :
        print('Process ' + self.name +  ' started.')
        with torch.no_grad() :
            
            white_render = White_Renderer(self.device, self.image_size)
            normal_render = Normal_Renderer(self.device, self.image_size)
            list_data = []
            count_main = 0
            dist = 2.75
            azim = 0.0
            elev = 0.0
            while(True) :
                count_main = count_main + 1
                if self.render_controller is not None :
                    if not self.render_controller.empty() :
                        control_pack = self.render_controller.get()
                        if control_pack['flag_para_change'] :
                            dist, azim, elev = control_pack['para']
                        elif control_pack['flag_save_obj'] :
                            print('control_pack[\'flag_save_obj\'] == True')
                            if v is not None :
                                f_copy = f.clone()
                                f_copy[:, 1] = f[:, 2]
                                f_copy[:, 2] = f[:, 1]
                                name_obj = 'recon_' + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.obj'
                                path_obj = os.path.join('./results/recon', name_obj)
                                write_obj(path=path_obj, verts=v.cpu().numpy(), faces=f_copy.cpu().numpy())
                                print('save obj, wrting done.')
                            else :
                                print('but v is None, cant save .obj now.')
                                None
                        elif control_pack['flag_to_animation'] :
                            if v is not None :
                                f_copy = f.clone()
                                f_copy[:, 1] = f[:, 2]
                                f_copy[:, 2] = f[:, 1]
                                name_obj = 'recon.obj'
                                path_obj = os.path.join('./results/latest', name_obj)
                                write_obj(path=path_obj, verts=v.cpu().numpy(), faces=f_copy.cpu().numpy())
                                print('to animation, wrting done.')
                            else :
                                print('but v is None, cant to animation now.')
                                None
                        elif control_pack['flag_render_normal_change'] :
                            self.flag_normal_shader = not self.flag_normal_shader
                        elif control_pack['flag_pause_change'] :
                            self.flag_pause = not self.flag_pause

                if self.flag_pause and count_main > 1 :
                    if v is not None :
                        if not self.flag_normal_shader :
                            image = white_render.render(
                                v=v, 
                                f=f, 
                                dist=dist,
                                azim=azim,
                                elev=elev
                            )
                            #image = (image + 1) * 0.5
                        else :
                            image = normal_render.render(
                                v=v, 
                                f=f, 
                                dist=dist,
                                azim=azim,
                                elev=elev
                            )
                            image = (image + 1) * 0.5
                    else :
                        image = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.uint8)
                    data_main = {'render': image}
                    data_out = {'main': data_main, 'vis':data_in['vis'], 'flag_save': False}
                    self.queue_next.put(data_out)
                    continue

                data_in_tmp = self.queue_prev.get()
                list_data.append(data_in_tmp)
                if len(list_data) < 4 :
                    continue
                else :
                    count = [0, 0, 0, 0]
                    for i in range(4) :
                        count[i] = list_data[i]['vis']['count']
                        min_count = min(count)
                    for i in range(4) :
                        if min_count == count[i] :
                            data_in = list_data[i]
                            del list_data[i]

                v = data_in['main']['verts']
                f = data_in['main']['faces']

                flag_save = False
                if v is not None :
                    if v.shape[0] > 10000 :
                        flag_save = True
                        #print('yeap')
    
                    if not self.flag_normal_shader :
                        image = white_render.render(
                            v=v, 
                            f=f, 
                            dist=dist,
                            azim=azim,
                            elev=elev
                        )
                        #image = (image + 1) * 0.5
                    else :
                        image = normal_render.render(
                            v=v, 
                            f=f, 
                            dist=dist,
                            azim=azim,
                            elev=elev
                        )
                        image = (image + 1) * 0.5
                    
                    def save_all(video_name = '103_for_unreal') :
                        if count_main % 3 == 0 :
                            path_save_stream_base = '/mnt/data/hj/c920pro/cap_result/'
                            video_name = video_name
                            path_save_stream = os.path.join(path_save_stream_base, video_name)
                            os.makedirs(path_save_stream, exist_ok=True)
                            path_vis_det = os.path.join(path_save_stream, 'input_' + str(count_main).zfill(4)) + '.png'
                            path_vis_seg = os.path.join(path_save_stream, 'seg_' + str(count_main).zfill(4)) + '.png'
                            path_vis_recon = os.path.join(path_save_stream, 'recon_' + str(count_main).zfill(4)) + '.obj'

                            img_det = cv.cvtColor(data_in['vis']['det'], cv.COLOR_RGB2BGR)
                            img_seg = cv.cvtColor(data_in['vis']['seg'], cv.COLOR_RGB2BGR)

                            cv.imwrite(path_vis_det, img_det)
                            cv.imwrite(path_vis_seg, img_seg)
                            f_copy = data_in['main']['faces'].clone()
                            v_copy = data_in['main']['verts'].clone()

                            # for unreal
                            #f_copy[:, 1] = f[:, 2]
                            #f_copy[:, 2] = f[:, 1]

                            v_copy[:, 2] = v[:, 1] + 1.0 #y to z
                            v_copy[:, 1] = v[:, 2] #z to y

                            v_copy = v_copy * 5.0
                            # for unreal done
                
                            write_obj(
                                path=path_vis_recon, 
                                verts=v_copy.cpu().numpy(),
                                faces=f_copy.cpu().numpy()
                            )
                            print('write a frame done.')
                    #save_all()
                    
                    
                    
                else :
                    image = torch.zeros((self.image_size, self.image_size, 3), dtype=torch.uint8)


                data_main = {'render': image}
                data_out = {'main': data_main, 'vis':data_in['vis'], 'flag_save':flag_save}
                self.queue_next.put(data_out)
            return

def write_obj(path, verts, faces) :
    with open(path, 'w') as f :
        for i in range(verts.shape[0]) :
            f.write('v %f %f %f\n' % (verts[i, 0], verts[i, 1], verts[i, 2]))

        faces_tmp = faces + 1
        for i in range(faces.shape[0]) :
            f.write('f %d %d %d\n' % (faces_tmp[i, 0], faces_tmp[i, 1], faces_tmp[i, 2]))


class White_Renderer() :
    def __init__(self, device, image_size) :
        self.device = device

        self.fx = 3.0
        self.fy = 3.0
        self.px = 0.0
        self.py = 0.0
        self.image_size=image_size

        self.light_positions = torch.zeros((1, 3), dtype=torch.float32).to(device)

        self.raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1 
        )
        
        self.materials = Materials(
            ambient_color=((1, 1, 1), ), 
            diffuse_color=((1, 1, 1), ), 
            specular_color=((1, 1, 1), ), 
            shininess=64, 
            device=self.device
        )
        
        self.blend_params = BlendParams(
            sigma=0.001, 
            gamma=0.001, 
            background_color=(1.0, 1.0, 1.0)
        )
        

    def render(self, v, f, dist=2.75, elev=0.0, azim=0.0) :
        mesh = Meshes(
            verts=[v.to(self.device)],   
            faces=[f.to(self.device)]
        )

        #verts_rgb = mesh.verts_normals_list()
        verts_rgb = torch.ones_like(mesh.verts_list()[0])[None]
        textures = TexturesVertex(verts_features=verts_rgb)

        mesh = Meshes(
            verts=mesh.verts_list(),   
            faces=mesh.faces_list(),
            textures=textures
        )

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, degrees=True) 

        x = np.sin(azim * np.pi / 180.0) * -1.0
        z = np.cos(azim * np.pi / 180.0) * -1.0
        y = np.sin(elev * np.pi / 180.0) * -1.0

        #print(azim, x, z)

        self.light_positions[0, 0] = x
        self.light_positions[0, 1] = y
        self.light_positions[0, 2] = z
        lights = DirectionalLights(device=self.device, direction=self.light_positions)


        cameras = PerspectiveCameras(
            device=self.device, 
            focal_length=((self.fx, self.fy),),
            principal_point=((self.px, self.py),),
            R=R, 
            T=T
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
                materials=self.materials,
                blend_params=self.blend_params
            )
        )

        #render
        images = renderer(mesh, cameras=cameras)
        image = images[0,:,:,0:3]
        return image



class Normal_Renderer() :
    def __init__(self, device, image_size) :
        self.device = device

        self.fx = 3.0
        self.fy = 3.0
        self.px = 0.0
        self.py = 0.0
        self.image_size=image_size

        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1, 1, 1))
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=np.log(1. / 1e-4 - 1.) * self.blend_params.sigma / 10, 
            faces_per_pixel=8, 
        )

    def render(self, v, f, dist=2.75, elev=0.0, azim=0.0) :
        f_copy = f.clone()
        f_copy[:, 1] = f[:, 2]
        f_copy[:, 2] = f[:, 1]
        f = f_copy

        mesh = Meshes(
            verts=[v.to(self.device)],   
            faces=[f.to(self.device)]
        )

        #verts_rgb = mesh.verts_normals_list()
        verts_rgb = torch.ones_like(mesh.verts_list()[0])[None]
        textures = TexturesVertex(verts_features=verts_rgb)

        mesh = Meshes(
            verts=mesh.verts_list(),   
            faces=mesh.faces_list(),
            textures=textures
        )

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, degrees=True) 

        x = np.sin(azim * np.pi / 180.0) * -1.0
        z = np.cos(azim * np.pi / 180.0) * -1.0
        y = np.sin(elev * np.pi / 180.0) * -1.0

        #print(azim, x, z)

        cameras = PerspectiveCameras(
            device=self.device, 
            focal_length=((self.fx, self.fy),),
            principal_point=((self.px, self.py),),
            R=R, 
            T=T
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=self.raster_settings
            ),
            shader=NormalShader(
                device=self.device, 
                cameras=cameras,
                blend_params=self.blend_params
            )
        )

        #render
        images = renderer(mesh, cameras=cameras)
        image = images[0,:,:,0:3]
        return image


if __name__ == '__main__' :
    device = torch.device('cuda:0')
    white_renderer = White_Renderer(device=device)
    v = torch.ones((100, 3), dtype=torch.float32, device=device)
    f = torch.ones((100, 3), dtype=torch.int, device=device)
    image = white_renderer.render(v=v, f=f)
