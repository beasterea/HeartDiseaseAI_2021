import argparse, os, datetime
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

import unet_model
import losses
from unet_model import UNet, UNetPP
from losses import BCEDiceLoss
from dataset import HeartDiseaseDataset
from metrics import iou_score
from utils import AverageMeter, str2bool

import pandas as pd

MODEL_NAMES = unet_model.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append("BCEWithLogitsLoss")

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default = '/content/drive/MyDrive/HeartDiseaseAI/DATA')

  # name of the model
  parser.add_argument('--name', default = None)

  # for the Net
  parser.add_argument('--batch_size', default = 8, type = int)
  parser.add_argument('--model', metavar = 'MODEL',default = 'UNetPP',choices = MODEL_NAMES,
                      help = 'model architecture : ' + '|'.join(MODEL_NAMES))
  parser.add_argument('--loss', default = 'BCEDiceLoss', choices = LOSS_NAMES,
                      help = 'loss function : ' + '|'.join(LOSS_NAMES))
  parser.add_argument('--deep_supervision', default = True, type = str2bool)
  parser.add_argument('--num_classes', default = 1, type = int, help = 'num of output channel')
  parser.add_argument('--threshold', default = 0.5, type = float)
  parser.add_argument('--input_channels', default = 3, type = int)
  parser.add_argument('--lr', default = 1e-3, type = float, metavar = 'LR', help = 'learning rate')
  parser.add_argument('--start_filter', default = 32, type = int)

  # scheduler for learning rate
  parser.add_argument('--scheduler', default = 'CosineAnnealingLR',
                      choices = ['ReduceLROnPlateau', 'CosineAnnealingLR', 'MultiStepLR', 'ConstantLR'])
  parser.add_argument('--min_lr', default = 1e-5, type = float)
  parser.add_argument('--patience', default = 5, type = int)
  

  # optimizer
  parser.add_argument('--epochs', default = 10, type = int)
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

    if config['deep_supervision']:
      outputs = net(input)
      loss = 0
      for output in outputs:
        loss += criterion(output, target, input_shape)
      loss /= len(outputs)
      ji = iou_score(outputs[-1], target, input_shape)
    
    else:
      output = net(input)
      loss = criterion(output, target, input_shape)
      ji = iou_score(output, target, input_shape)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_meters['loss'].update(loss.item(), input.size(0))
    avg_meters['JI'].update(ji, input.size(0))

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
      input = input.cuda()
      target = target.cuda()
      input_shape = info['img_shape']

          
      if config['deep_supervision']:
        outputs = net(input)
        loss = 0
        for output in outputs:
          loss += criterion(output, target, input_shape)
        loss /= len(outputs)
        ji = iou_score(outputs[-1], target, input_shape)
      
      else:
        output = net(input)
        loss = criterion(output, target, input_shape)
        ji = iou_score(output, target, input_shape)
    
      avg_meters['loss'].update(loss.item(), input.size(0))
      avg_meters['JI'].update(ji, input.size(0))

      postfix = OrderedDict([('loss' , avg_meters['loss'].avg),('JI' , avg_meters['JI'].avg)])
      pbar.set_postfix(postfix)
      pbar.update(1)
    pbar.close()

  return OrderedDict([('loss', avg_meters['loss'].avg), ('JI', avg_meters['JI'].avg)])


def main():
  now = datetime.datetime.now()
  config = vars(parse_args())

  # 만약에 best iou score을 기록한 모델이 생간다면 해당 모델의 checkpoint를 저장하기 위해 사용할 이름
  if config['name'] is None:
    if config['deep_supervision']:
      config['name'] = '%s_%s_deepsupervision' %(config['model'], str(now.strftime("%c")))
    else:
      config['name'] = '%s_%s_normal' %(config['model'], str(now.strftime("%c")))
  os.makedirs('models/%s' %config['name'], exist_ok = True)

  criterion = BCEDiceLoss().cuda()
  # configuration parameter입력값을 바탕으로 yaml파일로 저장함
  #with open('models/%s/config.yml' % config['name'], 'w') as f:
  #  yaml.dump(config, f)
  
   # create model
  print("=> creating model %s" % config['model'])
  if (config['model'] == 'ARUNet'):
    net = unet_model.__dict__[config['model']](config['num_classes'])
  else:
    net = unet_model.__dict__[config['model']](config['num_classes'],config['deep_supervision'],config['input_channels'])

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

  train_img_ids = list(set([os.path.splitext(os.path.basename(p))[0] for p in train_dirs]))
  valid_img_ids = list(set([os.path.splitext(os.path.basename(p))[0] for p in valid_dirs]))

  train_transform = Compose([
    transforms.Flip(),
    transforms.Normalize(),
  ])

  valid_transform = Compose([
    transforms.Normalize(),
  ])

  train_dataset = HeartDiseaseDataset(
      img_ids = train_img_ids,
      img_dir = os.path.join(config['dataset'], 'train', 'A2C'),
      mask_dir = os.path.join(config['dataset'], 'train', 'A2C'),
      num_classes = config['num_classes'],
      transform = train_transform
  )

  valid_dataset = HeartDiseaseDataset(
      img_ids = valid_img_ids,
      img_dir = os.path.join(config['dataset'], 'validation', 'A2C'),
      mask_dir = os.path.join(config['dataset'], 'validation', 'A2C'),
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
    ('epoch', []), ('loss', []), ('lr', []), ('JI', []), ('val_loss', []), ('val_JI', [])
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

    print('loss %.4f - JI %.4f - val_loss %.4f - val_JI %.4f'
              % (train_log['loss'], train_log['JI'], val_log['loss'], val_log['JI']))

    log['epoch'].append(epoch)
    log['lr'].append(config['lr'])
    log['loss'].append(train_log['loss'])
    log['JI'].append(train_log['JI'])
    log['val_loss'].append(val_log['loss'])
    log['val_JI'].append(val_log['JI'])

    pd.DataFrame(log).to_csv('models/%s/log.csv' %config['name'], index=False)

    trigger += 1

    if val_log['JI'] > best_iou:
      torch.save(net.state_dict(), 'models/%s/model.pth' %config['name'])
      best_iou = val_log['JI']
      print("=> saved best model")
      trigger = 0

    # early stopping
    # if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
    #   print("=> early stopping")
    #   break

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()