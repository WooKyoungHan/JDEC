import numpy as np
# from PIL import Image
import cv2
import os

import torch

from torchvision import transforms
import numpy as np
import models

from tqdm import tqdm
from utils_ import Averager

from einops.layers.torch import Rearrange
import utils.custom_transforms as ctrans
import dct_manip as dm
from utils_ import mkdir
import utils_



setname= 'LIVE1' #LIVE1 BSDS500 ICB 
if setname is 'LIVE1':
    data_path = './PATH_TO_LIVE1'
elif setname is 'BSDS500':
    data_path = './PATH_TO_B500'
elif setname is 'ICB':
    data_path = './PATH_TO_ICB'


model_path = './PATH_TO_MODEL'

model_spec = torch.load(model_path)['model']
model = models.make(model_spec, load_sd=True).cuda()

batch_y = Rearrange('c (h s1) (w s2) ph pw -> (h w) c s1 s2 ph pw',s1 = 140, s2=140)
batch_c = Rearrange('c (h s1) (w s2) ph pw -> (h w) c s1 s2 ph pw',s1 = 70, s2=70)

save = True
for i in [30]:
    preds=[]
    inputs = []
    res_psnr = Averager()
    res_psnr_b = Averager()
    res_ssim = Averager()
    inp_psnr = Averager()
    q = i
    print('-----'+str(q)+'-----')
    num = 0

    for item in tqdm(sorted(os.listdir(data_path))):
        img_png_ = cv2.imread(data_path+item,-1)
        h,w,_ = img_png_.shape
        gtgt_ = transforms.ToTensor()(cv2.cvtColor(img_png_,cv2.COLOR_BGR2RGB))
        size = 112*10

        img_png_ = np.concatenate([img_png_, np.flip(img_png_, [0])], 0)
        img_png_ = np.concatenate([img_png_, np.flip(img_png_, [1])], 1)
        img_png_ = np.concatenate([img_png_, np.flip(img_png_, [0])], 0)
        img_png_ = np.concatenate([img_png_, np.flip(img_png_, [1])], 1)[:size,:size,:]



        cv2.imwrite('./bin/temp_.jpg',img_png_,[int(cv2.IMWRITE_JPEG_QUALITY), q])

        input_ = dm.read_coefficients('./bin/temp_.jpg')
        inp_swin = input_[2]
        inp_swin_cbcr = input_[3]
        dqt_swin = input_[1]
        q_y = dqt_swin[0]
        q_cbcr = dqt_swin[1]
        inp_swin = torch.clamp(inp_swin * q_y,min=-1024,max=1016)
        inp_swin_cbcr = torch.clamp(inp_swin_cbcr * q_cbcr,min=-1024,max=1016)

        normalize = ctrans.ToRange(val_min=-1, val_max=1,orig_min=-1024, orig_max=1016)

        inp_swin = normalize(inp_swin)
        inp_swin_cbcr = normalize(inp_swin_cbcr)
        dqt_swin =torch.stack([q_y,q_cbcr],dim=0)
        dqt_swin = normalize(dqt_swin)

        with torch.no_grad():
            pred = model(inp_swin.unsqueeze(0).cuda(),inp_swin_cbcr.unsqueeze(0).cuda(),
            dqt_swin.unsqueeze(0).cuda())
            pred = pred.squeeze(0).detach().cpu() +0.5
        torch.cuda.empty_cache()

        inpinp = transforms.ToTensor()(cv2.cvtColor(cv2.imread('./bin/temp_.jpg'),cv2.COLOR_BGR2RGB))
        pred = pred[:,:h,:w]
        inpinp = inpinp[:,:h,:w]
        inputs.append(inpinp)
        preds.append(pred)

        inp_np=(inpinp*255).round().clamp(0,255).permute(1,2,0).numpy().astype(np.uint8)
        gt_np=(gtgt_*255).round().clamp(0,255).permute(1,2,0).numpy().astype(np.uint8)
        pred_np = (pred*255).round().clamp(0,255).permute(1,2,0).numpy().astype(np.uint8)

        if save is True:
            save_path = './bin//'+setname+'/' +str(q)
            mkdir(save_path)
            
            cv2.imwrite(save_path+'/'+item.split('.')[0]+'.png',
                        cv2.cvtColor(pred_np,cv2.COLOR_BGR2RGB))

        inp_psnr_temp = utils_.calculate_psnr(gt_np.copy(),inp_np.copy())
        psnr_temp = utils_.calculate_psnr(gt_np.copy(),pred_np.copy())
        psnrb_temp = utils_.calculate_psnrb(gt_np.copy(),pred_np.copy())
        ssim_temp = utils_.ssim_qg(transforms.ToTensor()(gt_np.copy()).unsqueeze(0),
        transforms.ToTensor()(pred_np.copy()).unsqueeze(0))

        inp_psnr.add(inp_psnr_temp)
        res_psnr.add(psnr_temp)
        res_psnr_b.add(psnrb_temp)

        res_ssim.add(ssim_temp.item())

    print('inp_psnr: {:.2f}'.format(inp_psnr.item()))
    print('------------------')
    print('Result_PSNR: {:.2f}'.format(res_psnr.item()))
    print('Result_PSNRB: {:.2f}'.format(res_psnr_b.item()))
    print('Result_SSIM: {:.3f}'.format(res_ssim.item()))