import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from lib.options import BaseOptions
from lib.mesh_util import *
from lib.sample_util import *
from lib.train_util import *
from lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm




# 注意：虽然能直接把网络结构也load进去，但是上面这些import不能少
# 否则见FQSplit使用另一种方案


model1 = torch.load("filter.pkl")
# 使用时把 model1 传进去
def stage1(model, images):
    with torch.no_grad(): #更安全
        feature_list , xx , yy = model(images)
        return feature_list[-1]




model2 = torch.load("mlp.pkl")
# 使用时把 model2 传进去
def stage2(model, points, calibs, im_feat):
    with torch.no_grad(): #更安全
        rot = calibs[:, :3, :3]
        trans = calibs[:, :3, 3:4]
        homo = torch.baddbmm(trans, rot, points)
        xy = homo[:, :2, :] / homo[:, 2:3, :]
        z = homo[:, 2:3, :] - trans[:, 2:3, :]

        in_img = (xy[:, 0] >= -1.0) & (xy[:, 0] <= 1.0) & (xy[:, 1] >= -1.0) & (xy[:, 1] <= 1.0)
        # 注意：以lib/model/DepthNormalizer.py里的实现为准，后续版本这里变了是 z_feat = z
        z_feat = z * (256 / 200 *1.28) 
        samples = torch.nn.functional.grid_sample(
            feat, 
            xy.transpose(1, 2).unsqueeze(2), 
            # 注意：以你lib/geometry.py里index函数的为准，后续版本我都是False
            align_corners=True
        ).squeeze(3)


        point_local_feat = torch.cat([samples, z_feat], 1)
        pred = in_img[:,None].float() * model(point_local_feat)
        return pred