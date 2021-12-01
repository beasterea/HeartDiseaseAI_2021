import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as transF

def crop_img(img, input_shape):
  h, w = input_shape[0], input_shape[1]

  dh = int((768 - h)/2)
  dw = int((1024 - w)/2)
  cropped = transF.crop(img, dw, dh, h, w)

  return cropped

def iou_score(predict, target, input_shape, crop = True):
  smooth = 1e-5
  intersection = 0
  union = 0
  n = predict.size(0) # batch size
  if crop:
    for i in range(n):
      predict_ = crop_img(predict[i], (input_shape[0][i], input_shape[1][i]))
      target_ = crop_img(target[i], (input_shape[0][i], input_shape[1][i]))
      if torch.is_tensor(predict_):
        predict_ = torch.sigmoid(predict_).data.cpu().numpy()
      if torch.is_tensor(target_):
        target_ = target_.data.cpu().numpy()
      predict_ = predict_ > 0.5
      target_ = target_ > 0.5
      
      intersection += (predict_ & target_).reshape(1, -1).sum(1)
      union += (predict_ | target_).reshape(1,-1).sum(1)
    return (intersection + smooth)/ (union+smooth)


  #predict = crop_img(predict, input_shape)

  if torch.is_tensor(predict):
    predict = torch.sigmoid(predict).data.cpu().numpy()
  if torch.is_tensor(target):
    target = target.data.cpu().numpy()
  predict_ = predict > 0.5
  target_ = target > 0.5
  intersection = (predict_ & target_).sum()
  union = (predict_ | target_).sum()

  return (intersection + smooth) / (union + smooth)

def dice_coef(predict, target, input_shape):
  smooth = 1e-5
  #predict = crop_img(predict, input_shape)

  predict = torch.sigmoid(predict).view(-1).data.cpu().numpy()
  target = target.view(-1).data.cpu().numpy()

  intersection = (predict * target).sum()

  return (2*intersection + smooth) / (predict.sum() + target.sum() + smooth)
  