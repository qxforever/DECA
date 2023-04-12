import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from tqdm import tqdm
from utils.renderer import SRenderY, set_rasterizer
from models.encoders import ResnetEncoder
from models.FLAME import FLAME, FLAMETex
from models.decoders import Generator
from utils import util
from utils.rotation_converter import batch_euler2axis
from utils.tensor_cropper import transform_points
from datasets import datasets
from utils.config import cfg
torch.backends.cudnn.benchmark = True


class DataGenerator():

    def __init__(self, config=None, device='cuda'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self.batch_size = self.cfg.gen.batch_size
        self.epoch = self.cfg.gen.epoch
        self.device = device
        self.flame = FLAME(self.cfg.model).to(self.device)
        self._setup_renderer(self.cfg.model)
        self._setup_parameter(self.cfg.model)

    def _setup_parameter(self, model_cfg):
        self.n_shape = model_cfg.n_shape
        self.n_tex = model_cfg.n_tex
        self.n_exp = model_cfg.n_exp
        self.n_pose = model_cfg.n_pose
        self.n_cam = model_cfg.n_cam
        self.n_light = model_cfg.n_light


    def _setup_renderer(self, model_cfg):
        set_rasterizer(self.cfg.rasterizer_type)
        self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size, rasterizer_type=self.cfg.rasterizer_type).to(self.device)
        # face mask for rendering details
        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        mask = imread(model_cfg.face_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # displacement correction
        fixed_dis = np.load(model_cfg.fixed_displacement_path)
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        mean_texture = imread(model_cfg.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start+int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict


    def rand_flame_parameter(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        radian = np.pi / 180.0
        codedict = {}
        codedict['shape'] = torch.randn(batch_size, self.n_shape, dtype=torch.float) * 0.6
        codedict['exp'] = torch.randn(batch_size, self.n_exp, dtype=torch.float) * 0.6
        pos1 = torch.randn(batch_size, 3, dtype=torch.float) * 7 * radian
        pos2 = torch.randn(batch_size, 3, dtype=torch.float) * 0.04
        codedict['pose'] = torch.cat([pos1, pos2], dim=1)
        codedict['cam'] = torch.zeros(batch_size, self.n_cam, dtype=torch.float)
        codedict['cam'][:, 0] = 10
        bias = torch.randn(batch_size, self.n_cam, dtype=torch.float)
        bias[:, 0] *= 1; bias[:, 1:] *= 0.01
        codedict['cam'] += bias

        for k, v in codedict.items():
            codedict[k] = v.to(self.device)
        
        codedict['path'] = []
        return codedict
    
    def generate_image(self, codedict):
        verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])
        trans_verts = util.batch_orth_proj(verts, codedict['cam']); 
        trans_verts[:,:,1:] = -trans_verts[:,:,1:]
        shape_images = self.render.render_shape(verts, trans_verts, h=self.image_size, w=self.image_size)
        return shape_images
    
    def test(self):
        filePath = './gen/flame-412'
        os.makedirs(filePath, exist_ok=True)
        cnt = len(os.listdir(filePath))
        codedicts = []
        for _ in tqdm(range(self.epoch)):
            codedict = self.rand_flame_parameter(batch_size=self.cfg.gen.batch_size)
            images = self.generate_image(codedict)
            for i in range(self.batch_size):
                filename = filePath + f'/{cnt}.jpg'
                codedict['path'].append(filename)
                cv2.imwrite(os.path.join(filename), util.tensor2image(images[i]))
                cnt = cnt + 1
            codedicts.append(codedict)
        np.save('./gen/412' + f'-{cnt}', codedicts)

if __name__ == '__main__':
    dg = DataGenerator()
    with torch.no_grad():
        dg.test()