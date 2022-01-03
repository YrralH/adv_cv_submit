import os
import sys
import time
import multiprocessing

#extenal lib
import numpy as np
import cv2 as cv
import ctypes as ct
import trimesh

#pytorch
import torch

class Process_Clean_Mesh(multiprocessing.Process) :
    def __init__(self, queue_prev, queue_next, name='') :
        super().__init__()
        self.name = name
        self.queue_prev = queue_prev
        self.queue_next = queue_next
        return

    def run(self) :
        print('Process ' + self.name + ' started.')
        with torch.no_grad() :
            engine = Clean_Mesh_Engine()
            while(True) :
                data_in = self.queue_prev.get()
                v = data_in['main']['verts']
                f = data_in['main']['faces']
                if v is not None :
                    v = v.numpy()
                    f = f.numpy()
                    mesh = trimesh.Trimesh(vertices =v, faces = f) 
                    v = mesh.vertices
                    f = mesh.faces
                    if v.shape[0] <= 10000000:
                        v, f = engine.process(v, f)
                    else:
                        print("guagua")
                    v = torch.from_numpy(v)
                    f = torch.from_numpy(f)
                else :
                    v = None
                    f = None
                data_main = {'verts': v, 'faces': f}
                data_out = {'main': data_main, 'vis':data_in['vis']}
                self.queue_next.put(data_out)
            return


class Clean_Mesh_Engine() :
    '''
    clean mesh in cpu
    '''
    def __init__(self) :
        self.so = ct.cdll.LoadLibrary("./wode.so")
        return

    def process(self, v, f) :
        v = np.ascontiguousarray(v.astype(np.float32))
        f = np.ascontiguousarray(f.astype(np.int32))
        pv = v.ctypes.data_as(ct.POINTER(ct.c_float))
        pf = f.ctypes.data_as(ct.POINTER(ct.c_int32))
        vsize = ct.c_int32(0)
        fsize = ct.c_int32(0)
        self.so.clean_mesh(pv,pf,v.shape[0],f.shape[0],ct.byref(vsize),ct.byref(fsize))
        vsize = vsize.value
        fsize = fsize.value
        v = v[0:vsize]
        f = f[0:fsize]
        return v,f
        '''
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        
        cc = mesh.split(only_watertight=False)
        
        if len(cc) > 0 :
            out_mesh = cc[0]
            bbox = out_mesh.bounds
            height = bbox[1,0] - bbox[0,0]
            for c in cc:
                bbox = c.bounds
                if height < bbox[1,0] - bbox[0,0]:
                    height = bbox[1,0] - bbox[0,0]
                    out_mesh = c

            v = torch.from_numpy(out_mesh.vertices).to(torch.float32)
            f = torch.from_numpy(out_mesh.faces)
            #print('spliting done')
        else :
            v = None
            f = None
            
        #v = torch.from_numpy(mesh.vertices).to(torch.float32)
        #f = torch.from_numpy(mesh.faces)
        return v, f
        '''