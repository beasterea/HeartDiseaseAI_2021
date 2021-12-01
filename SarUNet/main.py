import argparse, os
from collections import OrderedDict
from glob import glob

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from torch.optim import lr_scheduler
from tqdm import tqdm


from sarunet_model import HeartSarUnet
from losses import BCEDiceLoss
from losses import BCELoss
import sarunet_model
import losses
from dataset import HeartDiseaseDataset
from metrics import iou_score
from utils import AverageMeter, str2bool

import pandas as pd

MODEL_NAMES = sarunet_model.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append("BCEWithLogitsLoss")

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset',default = '/content/drive/MyDrive/HeartDiseaseAI/DATA', help = 'Base directory of the dataset')

  # name of the model
  parser.add_argument('--name', default = None)
  parser.add_argument('--all', default = True)
  parser.add_argument('--img_type', default = 'A2C')

  # for the Net
  parser.add_argument('--batch_size', default = 8, type = int)
  parser.add_argument('--input_channels', default = 3, type = int)
  parser.add_argument('--model', metavar = 'MODEL',default = 'UNetPP',choices = MODEL_NAMES,
                      help = 'model architecture : ' + '|'.join(MODEL_NAMES))
  parser.add_argument('--loss', default = 'BCELoss', choices = LOSS_NAMES,
                      help = 'loss function : ' + '|'.join(LOSS_NAMES))
  parser.add_argument('--num_classes', default = 1, type = int, help = 'num of output channel') # 최종 출력은 1개의 channel로
  parser.add_argument('--threshold', default = 0.5, type = float) # sigmoid를 취해준 값의 임계값
  parser.add_argument('--input_channel_num', default = 4, type = int) # input channel의 개수
  parser.add_argument('--lr', default = 1e-3, type = float, metavar = 'LR', help = 'learning rate') # 계속 변화를 줘야 하는 학습률

  # scheduler for learning rate
  parser.add_argument('--scheduler', default = 'CosineAnnealingLR',
                      choices = ['ReduceLROnPlateau', 'CosineAnnealingLR', 'MultiStepLR', 'ConstantLR'])
  parser.add_argument('--min_lr', default = 1e-5, type = float)
  parser.add_argument('--patience', default = 5, type = int)
  

  # optimizer
  parser.add_argument('--epochs', default = 20, type = int)
  parser.add_argument('--optimizer', default = 'Adam',
                      choices = ['Adam','SGD'], help = 'optimizers : '+ '|'.join(['Adam', 'SGD']))
  parser.add_argument('--momentum', default = '0.09', type = float)
  parser.add_argument('--weight_decay', default = 1e-4, type = float)
  config = parser.parse_args()

  return config

def train(net, train_loader, criterion, optimizer, config):
  """
  net : Net object for the task
  criterion : loss function
  optimizer : object used for optimization
  config : initial parameters parsed (dictionary type)
  """

  avg_meters = {'loss' : AverageMeter(), 'JI' : AverageMeter()}

  net.train()

  pbar = tqdm(total = len(train_loader)) #initialize the process visualization tool

  for input, target, info in train_loader:
    input = input.cuda()
    target = target.cuda()
    input_shape = info['img_shape']
    weights = info['weight'].cuda()

    output = net(input)
    if config['loss'] == 'BCELoss':
      loss = criterion(output, target, input_shape, weights)
    else:
      loss = criterion(output, target, input_shape)
    ji, jac = iou_score(output, target, input_shape)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_meters['loss'].update(loss.item(), input.size(0))
    avg_meters['JI'].update(jac, input.size(0))

    postfix = OrderedDict([('loss' , avg_meters['loss'].avg),('JI' , avg_meters['JI'].avg)])
    pbar.set_postfix(postfix)
    pbar.update()
  pbar.close()

  return OrderedDict([('loss', avg_meters['loss'].avg), ('JI', avg_meters['JI'].avg)])
  
def validate(net, valid_loader, criterion, config):
  avg_meters = {'loss': AverageMeter(),'JI': AverageMeter()}

  net.eval()

  with torch.no_grad():
    pbar = tqdm(total=len(valid_loader))
    for input, target, info in valid_loader:
      weights = info['weight'].cuda()
      input = input.cuda()
      target = target.cuda()
      input_shape = info['img_shape']

      output = net(input)
      if config['loss'] == 'BCELoss':
        loss = criterion(output, target, input_shape, weights)
      else:
        loss = criterion(output, target, input_shape)
      ji,jac = iou_score(output, target, input_shape)
      #print(f"loss:{loss} iou : {ji}")
      avg_meters['loss'].update(loss.item(), input.size(0))
      avg_meters['JI'].update(jac, input.size(0))

      postfix = OrderedDict([('loss' , avg_meters['loss'].avg),('JI' , avg_meters['JI'].avg)])
      pbar.set_postfix(postfix)
      pbar.update(1)
    pbar.close()

  return OrderedDict([('loss', avg_meters['loss'].avg), ('JI', avg_meters['JI'].avg)])


def main():
  config = vars(parse_args()) # parse_args()라는 training을 위해 설정해 놓은 값들
  if config['loss'] == 'BCELoss':
    criterion = BCELoss(crop = True).cuda()
  else:
    criterion = BCEDiceLoss(crop = True).cuda()
  net = HeartSarUnet(config['num_classes'], config['input_channels'])
  #net.load_state_dict(torch.load('/content/drive/MyDrive/HeartDiseaseAI/CODE_02/models/sarunet01.pth'))
  net = net.cuda()

  params = filter(lambda p:p.requires_grad, net.parameters())
  if (config['optimizer'] == 'Adam'):
    optimizer = optim.Adam(params, lr = config['lr'],)
  elif (config['optimizer'] == 'SGD'):
    optimizer = optim.SGD(params, lr = config['lr'], momentum = config['momentum'],)

  if config['scheduler'] == 'CosineAnnealingLR':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
  elif config['scheduler'] == 'ReduceLROnPlateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
  elif config['scheduler'] == 'MultiStepLR':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
  elif config['scheduler'] == 'ConstantLR':
    scheduler = None


  train_dirs = glob(os.path.join(config['dataset'], 'train', '*', '*')) # A2C, A4C image 모두 
  valid_dirs = glob(os.path.join(config['dataset'], 'validation', '*', '*'))

  train_img_dirs = list(filter(lambda x: x.split('/')[-1].split('.')[-1] == 'png', train_dirs))
  train_mask_dirs = list(filter(lambda x: x.split('/')[-1].split('.')[-1] == 'npy', train_dirs))
  valid_img_dirs = list(filter(lambda x: x.split('/')[-1].split('.')[-1] == 'png', valid_dirs))
  valid_mask_dirs = list(filter(lambda x: x.split('/')[-1].split('.')[-1] == 'npy', valid_dirs))

  train_img_ids = list(set([os.path.splitext(os.path.basename(p))[0] for p in train_dirs]))
  valid_img_ids = list(set([os.path.splitext(os.path.basename(p))[0] for p in valid_dirs]))

  train_transform = Compose([
    transforms.Flip(),
    transforms.Normalize(),
  ])

  valid_transform = Compose([
    transforms.Normalize(),
  ])

  if config['all'] == True:
    train_dataset = HeartDiseaseDataset(
      img_ids = None,
      img_dir = train_img_dirs,
      mask_dir = train_mask_dirs,
      num_classes = config['num_classes'],
      transform = train_transform
    )
    valid_dataset = HeartDiseaseDataset(
      img_ids = None,
      img_dir = valid_img_dirs,
      mask_dir = valid_mask_dirs,
      num_classes = config['num_classes'],
      transform = valid_transform
    )
  else:
    train_dataset = HeartDiseaseDataset(
      img_ids = train_img_ids,
      img_dir = os.path.join(config['dataset'], 'train', config['img_type']),
      mask_dir = os.path.join(config['dataset'], 'train', config['img_type']),
      num_classes = config['num_classes'],
      transform = train_transform
    )

    valid_dataset = HeartDiseaseDataset(
      img_ids = valid_img_ids,
      img_dir = os.path.join(config['dataset'], 'validation', config['img_type']),
      mask_dir = os.path.join(config['dataset'], 'validation', config['img_type']),
      num_classes = config['num_classes'],
      transform = valid_transform
    )

  train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size = config['batch_size'],
      shuffle = True,
      drop_last = True
  )

  valid_loader = torch.utils.data.DataLoader(
      valid_dataset, batch_size = config['batch_size'],
      shuffle = False, drop_last = False
  )

  log = OrderedDict([
    ('epoch', []), ('loss', []), ('lr', []), ('iou', []), ('valid_loss', []), ('valid_accuracy', [])
  ])

  best_iou = 0
  trigger = 0

  for epoch in range(config['epochs']):
    print(f"Epoch {epoch} / {config['epochs']}")

    # train for one epoch
    train_log = train(net,train_loader, criterion, optimizer, config)
    # validate for one epoch
    val_log = validate(net,valid_loader, criterion, config)

    if config['scheduler'] == 'CosineAnnealingLR':
      scheduler.step()
    elif config['scheduler'] == 'ReduceLROnPlateau':
      scheduler.step(val_log['loss'])

    print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['JI'], val_log['loss'], val_log['JI']))

    log['epoch'].append(epoch)
    log['lr'].append(config['lr'])
    log['loss'].append(train_log['loss'])
    log['iou'].append(train_log['JI'])
    log['valid_loss'].append(val_log['loss'])
    log['valid_accuracy'].append(val_log['JI'])

    # pd.DataFrame(log).to_csv('models/%s/log.csv' %config['name'], index=False)

    trigger += 1

    if val_log['JI'] > best_iou:
      torch.save(net.state_dict(), '/content/drive/MyDrive/HeartDiseaseAI/CODE_02/models/both_sarunetbceloss_01.pth')
      best_iou = val_log['JI']
      print("=> saved best model")
      trigger = 0

    # early stopping
    #if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
    #  print("=> early stopping")
    #  break

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()