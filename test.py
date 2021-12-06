from _typeshed import NoneType
import os, argparse
from glob import glob
import numpy as np

import cv2, torch
import torch.backends.cudnn as cudnn
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from tqdm import tqdm

from dataset import HeartDiseaseDataset
from metrics import iou_score, dice_coef
from utils import AverageMeter
from SarUNet.model import HeartSarUnet


MODEL_CKPT = os.curdir

def parse_args():
    parser = argparse.ArgumentParser()
    """First three arguments must not be changed"""
    parser.add_argument('--num_classes', default = 1)
    parser.add_argument('--input_channels', default = 1)
    parser.add_argument('--channel_in_start', default = 12)
    parser.add_argument('--pad', default = True)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    """User must choose the name of the model to save the result"""
    parser.add_argument('--name', default = None)
    """User must tell the directory of the input testing images"""
    parser.add_argument('--dataset', default = None, help = 'directory of the base datasets')
    parser.add_argument('--image_type', default = 'A2C')
    args = parser.parse_args()
    return args

def main():
    config = parse_args()

    # create model
    net = HeartSarUnet(config['num_classes'], config['input_channels'], config['channel_in_start'])
    net.load_state_dict(torch.load(MODEL_CKPT))
    net = net.cuda()
    net.eval()

    # load data
    test_dirs = sorted(os.path.join(config['dataset'], config['image_type'], '*'))
    test_img_dirs = sorted(list(filter(lambda x : x.split('/')[-1] == 'png', test_dirs)))
    test_mask_dirs = sorted(list(filter(lambda x : x.split('/')[-1] == 'npy', test_dirs)))

   
    test_dataset = HeartDiseaseDataset(
        img_ids = None,
        img_dir = test_img_dirs,
        mask_dir = test_mask_dirs,
        num_classes = config['num_classes'],
        pad = config['pad'],
        transform = None,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = 1,
        shuffle = False,
    )

    avg_meter = {'iou' : AverageMeter(), 'dice':AverageMeter()}

    with torch.no_grad():
        for input, target, info in tqdm(test_loader, total = len(test_loader)):
            input = input.cuda()
            target = target.cuda()
            img_shape = info['img_shape']

            # predict mask
            output = net(input)

            iou = iou_score(output, target, img_shape, crop = True)
            dice = dice_coef(output, target, img_shape)
            avg_meter['iou'].update(iou, input.size(0))
            avg_meter['dice'].update(dice, input.size(0))
            output = torch.sigmoid(output).cpu().numpy()

            img_id = list([os.path.splittext(os.path.basename(p))[0] for p in test_img_dirs])

            for i in range(len(output)):
                cv2.imwrite(os.path.join(config['dataset'], config['img_type'], img_id[i] + '.png'), 
                (output[i]>0.5).astype('uint8'))
    print('IoU : %.4f  Dice : %.4f' %avg_meter['iou'].avg %avg_meter['dice'].avg)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()




