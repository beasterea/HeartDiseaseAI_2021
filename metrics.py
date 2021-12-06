import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as transF
from sklearn.metrics import jaccard_score 
import warnings
warnings.filterwarnings('always')

def crop_img(img, input_shape):
  h, w = input_shape[0], input_shape[1]

  dh = int((768 - h)/2)
  dw = int((1024 - w)/2)
  cropped = transF.crop(img, dw, dh, h, w)

  return cropped

def iou_score(predict, target, input_shape, crop = True, sigmoid = False, softmax = False):
  smooth = 1e-5
  intersection = 0
  union = 0
  jac = 0
  n = predict.size(0) # batch size
  if softmax != True:
    for i in range(n):
      predict_ = crop_img(predict[i], (input_shape[0][i], input_shape[1][i]))
      target_ = crop_img(target[i], (input_shape[0][i], input_shape[1][i]))
      
      if torch.is_tensor(predict_):
        if sigmoid == True:
          predict_ = (predict_).data.cpu().numpy()
        else:
          predict_ = torch.sigmoid(predict_).data.cpu().numpy()
      if torch.is_tensor(target_):
        target_ = target_.data.cpu().numpy()
      predict_ = predict_ > 0.5
      target_ = target_ > 0.5

      jac += jaccard_score(predict_.reshape(1,-1)[0], target_.reshape(1,-1)[0])
      intersection += (predict_ & target_).sum((1,2))
      union += (predict_ | target_).sum((1,2))

      iou = (intersection + smooth) / (union + smooth)

      threshold = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    #print(f"intersection:{intersection} union:{union} jaccard : {jac/n}")
    return (intersection + smooth)/ (union+smooth), jac/n
  
  if crop == True or crop == 'True':
    for i in range(n):
      predict_ = crop_img(torch.softmax(predict[i],dim=0), (input_shape[0][i], input_shape[1][i]))
      target_ = crop_img(target[i], (input_shape[0][i], input_shape[1][i]))
      predict_ = torch.argmax(predict_, axis = 0, keepdim = True).float()
      
      if torch.is_tensor(predict_):
        if sigmoid == True:
          predict_ = predict_.data.cpu().numpy()
        else:
          predict_ = (predict_).data.cpu().numpy()
      if torch.is_tensor(target_):
        target_ = target_.data.cpu().numpy()
      predict_ = predict_ > 0.5
      target_ = target_ > 0.5

      jac += jaccard_score(predict_.reshape(1,-1)[0], target_.reshape(1,-1)[0])
      intersection += (predict_ & target_).sum((1,2))
      union += (predict_ | target_).sum((1,2))

      iou = (intersection + smooth) / (union + smooth)

      threshold = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    #print(f"intersection:{intersection} union:{union} jaccard : {jac/n}")
    return (intersection + smooth)/ (union+smooth)


  #predict = crop_img(predict, input_shape)


  if torch.is_tensor(predict):
    if sigmoid == True:
      predict_ = predict_.data.cpu().numpy()
    else:
      predict_ = torch.sigmoid(predict_).data.cpu().numpy()
  if torch.is_tensor(target):
    target = target.data.cpu().numpy()
  predict_ = predict > 0.5
  target_ = target > 0.5
  #jac = jaccard_score(predict.reshape(1,-1)[0], target.reshape(1,-1)[0])
  intersection = (predict_ & target_).sum()
  union = (predict_ | target_).sum()

  return (intersection + smooth) / (union + smooth), jac/n


def dice_coef(predict, target, input_shape):
  smooth = 1e-5
  n = predict.size(0)
  dice=  0
  for i in range(n):
    predict_ = crop_img(predict[i], (input_shape[0][i], input_shape[1][i]))[0]
    target_ = crop_img(target[i], (input_shape[0][i], input_shape[1][i]))[0]

    predict_ = torch.sigmoid(predict_).view(-1).data.cpu().numpy()
    target_ = target_.clone().detach()
    target_ = target_.view(-1).data.cpu().numpy()

    intersection = (predict_ * target_).sum()
    dice += (2*intersection + smooth) / (predict_.sum() + target_.sum() + smooth)
  return dice / n


  