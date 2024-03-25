# modified from: https://github.com/yinboc/liif

import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils_
from utils_ import createDirectory

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))

    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers=8, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if os.path.exists(config.get('resume')):
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils_.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        # for _ in range(epoch_start - 1):
        #     lr_scheduler.step()
        print(epoch_start)
        print(optimizer.param_groups[0]['lr'])
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils_.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils_.compute_num_params(model, text=True)))
    # print(model)
    return model, optimizer, epoch_start, lr_scheduler



def train(train_loader, model, optimizer, \
         epoch):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils_.Averager()
    
    num_dataset = 3450
    iter_per_epoch = int(num_dataset / config.get('train_dataset')['batch_size'] \
                        * config.get('train_dataset')['dataset']['args']['repeat'])
    iteration = 0
    pbar = tqdm(train_loader, leave=False, desc='train')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        dct = batch['inp']
        q_map = batch['dqt']

        chroma = batch['chroma']
        gt_dct = batch['gt']
        pred_dct = model(dct,chroma,q_map)

        loss = loss_fn(pred_dct,gt_dct)

        # tensorboard
        writer.add_scalars('loss', {'train': loss.item()}, (epoch-1)*iter_per_epoch + iteration)
        iteration += 1
        
        train_loss.add(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description('loss {:.4f}'.format(loss.item()))
        pred_dct = None; loss = None
        
    return train_loss.item()

def valid(valid_loader, model, epoch):
    model.eval()
    valid_psnr = utils_.Averager()
    metric_fn = utils_.calc_psnr

    
    num_dataset = 10
    iter_per_epoch = int(num_dataset / config.get('val_dataset')['batch_size'] \
                        * config.get('val_dataset')['dataset']['args']['repeat'])
    iteration = 0
    for batch in tqdm(valid_loader, leave=False, desc='valid'):
        for k, v in batch.items():
            batch[k] = v.cuda()

        with torch.no_grad():
            dct = batch['inp']
            q_map = batch['dqt']

            chroma = batch['chroma']
            gt_dct = batch['gt']
            pred_dct = model(dct,chroma,q_map)
            psnr = metric_fn(gt_dct,pred_dct)
            

        
        # tensorboard
        writer.add_scalars('psnr', {'valid': psnr}, (epoch-1)*iter_per_epoch + iteration)
        iteration += 1
        
        valid_psnr.add(psnr.item())
        pred = None; loss = None
        
    return valid_psnr.item()


def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils_.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()


    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils_.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer, epoch)
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.5f}'.format(train_loss))

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))
        

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = valid(val_loader, model_,epoch)

            log_info.append('val: psnr={:.5f}'.format(val_res))

            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils_.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils_.time_text(t), utils_.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--folder', default=None)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)
    
    if args.folder is not None:
        save_folder = './save/' + args.folder
        save_path = os.path.join(save_folder, save_name)
        createDirectory(save_path)
    
    
    main(config, save_path)