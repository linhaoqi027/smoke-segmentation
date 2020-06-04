import cv2
import numpy as np
from PIL import Image
import os

ckpt_result = './vis/deform_result' # the directory of checkpoints.
if not os.path.exists(ckpt_result):
    os.makedirs(ckpt_result)

ckpt_label = './my_gray17_vis/label' # the directory of checkpoints.
if not os.path.exists(ckpt_label):
    os.makedirs(ckpt_label)

for idx in range(1,300):
    img = '/home/ecust/lhq/smoke/testing_data/blendall/'+str(idx)+'.png'
    mask = '/home/ecust/lhq/smoke/result_deform/'+str(idx)+'.png'
    img = Image.open(img).convert('RGBA')
    img=img.resize((256,256))
    mask = Image.open(mask).convert('RGBA')
    mask = np.array(mask)
    mask[:,:,2]=0
    mask = Image.fromarray(mask.astype('uint8')).convert('RGBA')
    #生成掩膜图
    result = Image.blend(img, mask, 0.2)
    #保存生成的掩膜图
    result.save(ckpt_result+'/'+str(idx)+'.png')

    # img = '/home/ecust/lhq/smoke/17/pic/'+str(idx)+'.png'
    # mask = '/home/ecust/lhq/smoke/17/cv2_mask/'+str(idx)+'.png'
    # img = Image.open(img).convert('RGBA')
    # img = img.resize((256, 256))
    # mask = Image.open(mask).convert('RGBA')
    # mask = mask.resize((256, 256))
    # mask = np.array(mask)
    # mask[:,:,2]=0
    # mask = Image.fromarray(mask.astype('uint8')).convert('RGBA')
    # #生成掩膜图
    # label = Image.blend(img, mask, 0.3)
    # #保存生成的掩膜图
    # label.save(ckpt_label+'/'+str(idx)+'.png')

