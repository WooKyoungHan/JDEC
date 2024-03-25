import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import register

import cv2

import utils.custom_transforms as ctrans

    



@register('JDEC-decoder_toimage_rgb')
class JDEC_decoder_datawrapper(Dataset):
    def __init__(self, dataset,augment=False, inp_size=None,valid = False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.valid = valid
        self.normalize = ctrans.ToRange(val_min=-1, val_max=1,orig_min=-1024, orig_max=1016)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        input_, gt_ = self.dataset[idx]
        quant_table = input_[1]
        q_y = quant_table[0]
        q_cbcr = quant_table[1]
        Y =input_[2]
        cbcr = input_[3]

        Y_gt =gt_

        width = self.inp_size
        height = self.inp_size

        x0 = random.randint(0, Y_gt.shape[0]//16 - (height*8)//16 -1)
        y0 = random.randint(0, Y_gt.shape[1]//16 - (width*8)//16 -1)

        Y = Y[:, 2*x0:2*x0 + height, 2*y0: 2*y0 + width, :,:]
        cbcr = cbcr[:, x0:x0 + height//2, y0: y0 + width//2, :,:]
        Y_gt = Y_gt[16*x0:16*x0 + 8*height, 16*y0: 16*y0 + 8*width,:]

        Y_gt = cv2.cvtColor(Y_gt,cv2.COLOR_BGR2RGB)
        Y_gt = transforms.ToTensor()(Y_gt)#.unsqueeze(0)
        Y_gt = Y_gt - 0.5

        Y = torch.clamp(Y * q_y, min=-2**10, max=2**10-8)
        cbcr = torch.clamp(cbcr * q_cbcr,min=-2**10, max=2**10-8)
        Y =self.normalize(Y)
        cbcr =self.normalize(cbcr)
        quant_table =torch.stack([q_y,q_cbcr],dim=0)
        quant_table = self.normalize(quant_table)
        return {
            'inp': Y,
            'chroma':cbcr,
            'dqt': quant_table,
            'gt': Y_gt
        }
    
