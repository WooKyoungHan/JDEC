import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils_ import make_coord

import numpy as np 


@register('jdec')
class JDEC(nn.Module):
    def __init__(self, encoder_spec,decoder_spec=None , hidden_dim=256):
        super().__init__()  
        self.encoder_dct = models.make(encoder_spec)
        self.coef = nn.Sequential(*[nn.Conv2d(self.encoder_dct.out_dim, hidden_dim, 3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(hidden_dim, hidden_dim, 3,stride=1,padding=1)])
        self.freq = nn.Sequential(*[nn.Conv2d(self.encoder_dct.out_dim, hidden_dim*2, 3,stride=1,padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(hidden_dim*2, hidden_dim*2, 3,stride=1,padding=1)])

        self.qmapp = nn.Conv2d(128, hidden_dim, 1,stride=1,padding=0)
        
        self.dequantizator=models.make(decoder_spec,args={'in_dim': hidden_dim,'out_dim':3}) 


    def de_quantization(self,dct,cbcr,qmap):
        bs,h,w =dct.shape[0], dct.shape[2],dct.shape[3]
        ih=h*8
        iw=w*8
        coord = make_coord([ih,iw],flatten=False).permute(2,0,1).cuda()#2,H,W 

        coord[0,:,:] = (2*(((coord[0,:,:]*ih+ih-1)/2)%4)+1)/8
        coord[1,:,:] = (2*(((coord[1,:,:]*iw+iw-1)/2)%4)+1)/8

        self.feat = self.encoder_dct(dct,cbcr)  
        
        self.feat = self.feat.reshape(bs,2*h,2*w,-1)
        self.feat = self.feat.permute(0,3,1,2)
        self.freqq = self.freq(self.feat)

        self.coeff = self.coef(self.feat)
        self.basis = self.qmapp(qmap.reshape(bs,-1,1,1))
        self.coeff = self.basis * self.coeff
        

        self.freq_1 ,self.freq_2= self.freqq.split(self.freqq.shape[1]//2,dim=1)
        self.freq_1  = torch.cos(np.pi*(self.freq_1[:,:,:,None,:,None])*(coord[0,:,:].unsqueeze(0).reshape(1,h*2,4,w*2,4)[None,:,:,:,:,:]))
        self.freq_2  = torch.cos(np.pi*(self.freq_2[:,:,:,None,:,None])*(coord[1,:,:].unsqueeze(0).reshape(1,h*2,4,w*2,4)[None,:,:,:,:,:]))
        inp = self.freq_1*self.freq_2*self.coeff[:,:,:,None,:,None]
        inp = inp.reshape(bs,-1,ih,iw)
        pred = self.dequantizator(inp)
        return pred  


    def forward(self, dctt,qmapp,cbcr):
        pred = self.de_quantization(dctt,qmapp,cbcr)
        return pred
