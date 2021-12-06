import os, argparse, warnings
from glob import glob
import numpy as np

import cv2, torch
import torch.backends.cudnn as cudnn
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from tqdm import tqdm
from PIL import Image

from dataset import HeartDiseaseDataset
from metrics import iou_score, dice_coef, crop_img
from utils import AverageMeter
from sarunet_model import HeartSarUnet


#MODEL_CKPT = '/content/drive/MyDrive/HeartDiseaseAI/CODE_02/models/crop_sarunet_sigmoid_noflip_bce01.pth'
MODEL_CKPT = ''
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
    parser.add_argument('--data_path', default = None)
    parser.add_argument('--data_type', default = None, help = 'validation or test')
    parser.add_argument('--image_type', default = 'A2C')
    args = parser.parse_args()
    return args

def main():
    config = vars(parse_args())
    if config['image_type'] == 'A2C':
        MODEL_CKPT = os.path.join(os.getcwd(),'models', 'crop_sarunet_A2C_bce01.pth')
    else:
        MODEL_CKPT = os.path.join(os.getcwd(),'models', 'sarunet_A4C.pth')

    # create model
    net = HeartSarUnet(config['num_classes'], config['input_channels'], config['channel_in_start'])
    net.load_state_dict(torch.load(MODEL_CKPT))
    net = net.cuda()
    net.eval()

    # load data
    test_dirs = sorted(glob(os.path.join(config['data_path'], config['data_type'], config['image_type'], '*')))
    test_img_dirs = sorted(list(filter(lambda x: x.split('/')[-1].split('.')[-1] == 'png', test_dirs)))
    test_mask_dirs = sorted(list(filter(lambda x: x.split('/')[-1].split('.')[-1] == 'npy', test_dirs)))

   
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
        idx = 0
        img_id = list([os.path.splitext(os.path.basename(p))[0] for p in test_img_dirs])
        for input, target, info in tqdm(test_loader, total = len(test_loader)):
            input = input.cuda()
            target = target.cuda()
            img_shape = info['img_shape']

            # predict mask
            output = net(input)

            iou = iou_score(output, target, img_shape, crop = True)
            dice = dice_coef(output, target, img_shape)
            avg_meter['iou'].update(iou[-1], input.size(0))
            avg_meter['dice'].update(dice, input.size(0))
            output_ = crop_img(output, (img_shape[0][0], img_shape[1][0]))
            output_ = torch.sigmoid(output_).cpu().numpy()
            
            for i in range(len(output)):
                output_[i][0] = output_[i][0] > 0.5
                cv2.imwrite(os.path.join(config['data_path'] , config['data_type'], config['image_type'], 'result',img_id[idx] + '.png'), 
                (output_[i][0]).astype(float)*255)

                np.save(os.path.join(config['data_path'] , config['data_type'], config['image_type'], 'result',img_id[idx] + '.npy'),
                output_[i][0].astype('uint8'))
                idx += 1
    
    dice = avg_meter['dice'].avg
    print(f'IoU : {dice/(2-dice)}')
    print('JI : %.4f  Dice : %.4f' %(avg_meter['iou'].avg, avg_meter['dice'].avg))


    torch.cuda.empty_cache()

if __name__ == '__main__':
    warnings.filterwarnings(action = 'ignore')
    main()




