import os
import json

import pickle
from torch.utils.data import Dataset
from datasets import register
import cv2

import random

import dct_manip as dm
        
        
@register('image-folder-png')
class ImageFolderPNG(Dataset):
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    #print(file)
                    with open(bin_file, 'wb') as f:
                        print(file)
#                         print(cv2.imread(file,-1))
                        pickle.dump(cv2.imread(file),f)
                    print('dump', bin_file)
                self.files.append(bin_file)
                self.bin_root = bin_root
            elif cache == 'in_memory':
                print(file)
                self.files.append(cv2.imread(file))
    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return cv2.imread(x)

        elif self.cache == 'bin':
            #os.system('rm '+self.bin_root+'.pkl')
            with open(x, 'rb') as f:
                x = pickle.load(f)
            return x

        elif self.cache == 'in_memory':
            return x
        
        
@register('image-folder-embed-image')
class ImageFolderJPEG_embed_image(Dataset):
    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        root_path_ = root_path
        if split_file is None:
            filenames = sorted(os.listdir(root_path_))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]
        
    
        self.files = []

        for filename in filenames:
            file = os.path.join(root_path_, filename)
#             print(file)
            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path_),
                    '_bin_' + os.path.basename(root_path_))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    #print(file)
                    with open(bin_file, 'wb') as f:
                        print(file)
                        #  
                        # temp_file = dm.read_coefficients(file)
                        data = dm.read_coefficients(file)
                        data = list(data)
                        # 0: dim, 1: quant tables, 2: Luma, 3: Chroma, 3: image
                        data.append(cv2.imread(file,-1))
                        pickle.dump(data,f)
                    print('dump', bin_file)
                self.files.append(bin_file)
                self.bin_root = bin_root
            elif cache == 'in_memory':
                # print(file)
                data = dm.read_coefficients(file)
                data = list(data)
                # 0: dim, 1: quant tables, 2: Luma, 3: Chroma, 3: image
                data.append(cv2.imread(file,-1))
                self.files.append(data)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            data = dm.read_coefficients(x)
            data = list(data)
            # 0: dim, 1: quant tables, 2: Luma, 3: Chroma, 3: image
            data.append(cv2.imread(x,-1))            
            return data

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            return x

        elif self.cache == 'in_memory':
            return x
        

@register('train-paired-imageset')
class PairedImageFolders(Dataset):
    def __init__(self, root_path_inp,root_path_gt, **kwargs):
        self.root_path_inp = root_path_inp
        self.inp_quality = [0,10,20,30,40,50,60,70,80,90,100]

        self.dataset_jpeg10 = ImageFolderJPEG_embed_image(self.root_path_inp+ '_'+str(self.inp_quality[1]), **kwargs)#10
        self.dataset_jpeg20 = ImageFolderJPEG_embed_image(self.root_path_inp+ '_'+str(self.inp_quality[2]), **kwargs)#
        self.dataset_jpeg30 = ImageFolderJPEG_embed_image(self.root_path_inp+ '_'+str(self.inp_quality[3]), **kwargs)#
        self.dataset_jpeg40 = ImageFolderJPEG_embed_image(self.root_path_inp+ '_'+str(self.inp_quality[4]), **kwargs)#
        self.dataset_jpeg50 = ImageFolderJPEG_embed_image(self.root_path_inp+ '_'+str(self.inp_quality[5]), **kwargs)#
        self.dataset_jpeg60 = ImageFolderJPEG_embed_image(self.root_path_inp+ '_'+str(self.inp_quality[6]), **kwargs)#
        self.dataset_jpeg70 = ImageFolderJPEG_embed_image(self.root_path_inp+ '_'+str(self.inp_quality[7]), **kwargs)#
        self.dataset_jpeg80 = ImageFolderJPEG_embed_image(self.root_path_inp+ '_'+str(self.inp_quality[8]), **kwargs)#
        self.dataset_jpeg90 = ImageFolderJPEG_embed_image(self.root_path_inp+ '_'+str(self.inp_quality[9]), **kwargs)#
        self.dataset_jpeg100 = ImageFolderJPEG_embed_image(self.root_path_inp+ '_'+str(self.inp_quality[10]), **kwargs)#10

        self.dataset = ImageFolderPNG(root_path_gt, **kwargs)#100

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inp_quality = random.choice([10,20,30,40,50,60,70,80,90,100])
        if inp_quality == 10:
            self.dataset_2 = self.dataset_jpeg10 
        elif inp_quality == 20:
            self.dataset_2 = self.dataset_jpeg20 
        elif inp_quality == 30:
            self.dataset_2 = self.dataset_jpeg30 
        elif inp_quality == 40:
            self.dataset_2 = self.dataset_jpeg40 
        elif inp_quality == 50:
            self.dataset_2 = self.dataset_jpeg50 
        elif inp_quality == 60:
            self.dataset_2 = self.dataset_jpeg60 
        elif inp_quality == 70:
            self.dataset_2 = self.dataset_jpeg70 
        elif inp_quality == 80:
            self.dataset_2 = self.dataset_jpeg80 
        elif inp_quality == 90:
            self.dataset_2 = self.dataset_jpeg90  
        elif inp_quality == 100:
            self.dataset_2 = self.dataset_jpeg100 
        return self.dataset_2[idx], self.dataset[idx]
    
@register('valid-paired-dataset')
class PairedImageFolders(Dataset):
    def __init__(self,root_path_inp,root_path_gt, **kwargs):
        self.dataset_2 = ImageFolderJPEG_embed_image(root_path_inp, **kwargs)#10
        self.dataset = ImageFolderPNG(root_path_gt, **kwargs)#100

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return  self.dataset_2[idx], self.dataset[idx]

