import torch, cv2, os
import numpy as np

def make_mask(img_id, mask_id, root_dir, ):
  """
  Args :
    img_id : idx number of the file(확장자를 포함한 파일명)
    mask_id : 위와 동일
    root_dir : root directory of the data to use for training
  Output :
    1. None, but saves the masked heart image to the directory
    2. Masked image
  """
  img_dir = os.path.join(root_dir, img_id)
  mask_dir = os.path.join(root_dir, mask_id)

  img = cv2.imread(img_dir)
  mask = np.load(mask_dir)

  shape = img.shape
  output = np.zeros(shape = shape)

  if shape[-1] == 3:
    for i in range(3):
      output[:,:,i][np.where(mask == 1)] = 1
  else:
    output[np.where(mask == 1)] = 1

  # plt.imshow(output)
  return output

  


def make_bbox(img_id, mask_id, root_dir, ):
  """
  실제 원본 초음파 이미지에서 좌심실 부분의 ROI를 찾는데에 사용할 수 있다.
  Args : 
    img_id : idx number of the file (경로를 만들기 위해서 확장자까지 포함한 파일 이름을 사용)
    mask_id : 위와 마찬가지
    root_dir : 파일을 저장하는 root directory
  
  """
  mask = make_mask(img_id, mask_id, root_dir)[:,:,0]
  shape = mask.shape

  min_w, max_w, min_h, max_h = 2000, 0, 2000, 0
  start = 0
  for i in range(shape[0]): # 세로 행의 개수만큼
    x_list = np.where(mask[i,:]==1)[0]
    if x_list.size == 0:
      if start == 1:
        max_h = i
        start = 2
      continue
    else:
      if start == 0:
        min_h = i
        start = 1
      x_min, x_max = np.min(x_list), np.max(x_list)
      if x_min < min_w:
        min_w = x_min
      if x_max > max_w:
        max_w = x_max
     
  img_dir = os.path.join(root_dir, img_id)
  img = cv2.imread(img_dir,cv2.IMREAD_GRAYSCALE)
  img = 254-img
  ROI = img[min_h:max_h, min_w:max_w]

  mean, STD = cv2.meanStdDev(ROI)
  offset = 0.2
  clipped = np.clip(img, mean-offset*STD, mean + offset*STD).astype(np.uint8)
  result = cv2.normalize(clipped, clipped, 0, 255, norm_type = cv2.NORM_MINMAX)

  
  img = cv2.rectangle(img, pt1= (int(min_w), int(min_h)),pt2= (int(max_w), int(max_h)), color = (255,0,0), thickness = 5)

  return ROI, result
class HeartDiseaseDataset(torch.utils.data.Dataset):
  def __init__(self, img_ids, img_dir, mask_dir, num_classes, pad = True,transform = None, gray = True):

    self.img_ids = img_ids
    self.img_dir = img_dir
    self.mask_dir = mask_dir
    self.num_classes = num_classes
    self.transform = transform
    self.gray = gray
    self.pad = pad

  def _pad_img(self, img):
    shape = img.shape

    w,h = shape[1], shape[0]

    if (shape[0] != 768 or shape[1] != 1024):
      new_img = np.zeros(shape = (768, 1024,shape[-1]), dtype = np.uint8)
    else:
      new_img = img
    
    dh = int((768 - shape[0])/2)
    dw = int((1024 - shape[1])/2)

    #new_img[:,:,-1] = 255 # 마지막 alpha channel값은 255로
    new_img[dh:dh+h,dw:dw+w,:] = img[:,:,:]

    return new_img
  def _change_top(self, img):
    """insert the image and change the top part into black"""
    shape = img.shape
    for i in range(shape[0]):
      if (img[i][0] != 0):
        if img[i][0] == 1:
          break
        else:
          img[i, :] = 0
      else:
        break
    return img

  
  def _calc_weights(self, mask):
    h,w = mask.shape[0], mask.shape[1]
    entire = h*w
    answer = mask.reshape(1,-1).sum(1)
    weight = (entire - answer) / answer
    #print(weight)
    
    return torch.tensor(weight)
    


  def __len__(self):
    return len(self.img_dir)

  def __getitem__(self, idx):
    """
    idx (int) : numbers of the index for the image id in the list
    """
    if self.img_ids is None:
      if self.gray:
        img = cv2.imread(self.img_dir[idx], cv2.IMREAD_GRAYSCALE)
        img = self._change_top(img)
        img = np.expand_dims(img, axis = -1)
        mask = np.load(self.mask_dir[idx])
      else:
        img = cv2.imread(self.img_dir[idx])
        img = self._change_top(img)
        mask = np.load(self.mask_dir[idx])

    else:
      if self.gray:
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id+'.png'), cv2.IMREAD_GRAYSCALE)
        img = self._change_top(img)
        img = np.expand_dims(img, axis = -1)
        mask = np.load(os.path.join(self.mask_dir, img_id+'.npy'))
      else:
        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id+'.png'))
        img = self._change_top(img)
        mask = np.load(os.path.join(self.mask_dir, img_id+'.npy'))
    img_shape = img.shape[:-1] # (W,H)
    weights = self._calc_weights(mask)
    if self.pad:
      img = self._pad_img(img)
      mask = np.expand_dims(mask, axis = -1)
      mask = self._pad_img(mask)
    else:
      mask = np.expand_dims(mask, axis = -1)
    #print(mask.shape)
  

    if self.transform is not None:
      transformed = self.transform(image = img, mask = mask)
      img = transformed['image']
      mask = transformed['mask']

    from albumentations.augmentations import transforms
    from albumentations.core.composition import Compose
    """
    normalization
    #img_transform = Compose([transforms.Normalize(mean = 48, std = 38, max_pixel_value = 255.0)])
    img_transform = Compose([transforms.Normalize()])
    normed_img = img_transform(image = img, mask = mask)['image']
    """
    """
    transforms
    img_transform = Conpose(OneOf([
      transforms.ShiftScaleRotate(),
      transforms.GaussNoise()
    ]))
    img = img_transform(image = img, mask = mask)['image']
    """
  
    #img = normed_img.astype('float32')/255# 0-255사이의 uint8 -> 0-1 사이의 float32
    if np.max(img) > 1.0:
      img = img.astype('float32')/255
    else:
      img = img.astype('float32')
    mask = mask.astype('float32')
    #print(f"mask : {mask}")
    if len(img.shape) == 2:
      img = img.expand_dims(img, axis = -1)
    img = img.transpose(2, 0, 1) # (3, 434, 636) 또는 (3, 600, 800)

    mask = mask.transpose(2, 0, 1)

    return img, mask, {'img_id' : self.img_dir, 'img_shape':img_shape, 'weight' : weights}
