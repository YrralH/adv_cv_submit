import sys
import os

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from lib.PIFu.lib.options import BaseOptions
from lib.PIFu.lib.mesh_util import *
from lib.PIFu.lib.sample_util import *
from lib.PIFu.lib.train_util import *
from lib.PIFu.lib.model import *

from PIL import Image
import torchvision.transforms as transforms
import glob
import tqdm

# 如下调用：就是保证opt一致
# 我是放在apps里做的
'''
python ./lib/PIFu/apps/FQSplit.py --batch_size 1 \
    --mlp_dim 257 1024 512 256 128 1 \
    --num_stack 4 \
    --num_hourglass 2 \
    --hg_down 'ave_pool' \
    --norm 'group' \
    --norm_color 'group' 
'''

opt = BaseOptions().parse()
cpu = torch.device('cpu')
netG = HGPIFuNet(opt, 'perspective').to(device=cpu)
# 你的ckpt路径
netG.load_state_dict(torch.load('./lib/PIFu/checkpoints/example/netG_latest', map_location=cpu))
netG.eval()
a = netG.image_filter
b = netG.surface_classifier
torch.save(a, './lib/PIFu/checkpoints/filter.pkl')
torch.save(b, './lib/PIFu/checkpoints/mlp.pkl')
'''
如上所示，你可以选择save state_dict()
像下面这样创建模型，然后再load
a = HGFilter(opt)
b = SurfaceClassifier(
        filter_channels=opt.mlp_dim,
        num_views=opt.num_views,
        no_residual=opt.no_residual,
        last_op=nn.Sigmoid())
'''