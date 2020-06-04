# Loading the checkpoints for testing
from dataset import TestDataset
from torch.utils.data import DataLoader
import os
import torch
from DSS import DSS
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import PIL
import math
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix


ckpt_path = 'saved_models/105_checkpoint.pth.tar'


model = DSS()
model = torch.nn.DataParallel(model, device_ids=list(range(1))).cuda()
cudnn.benchmark = True

if os.path.isfile(ckpt_path):
    print("=> Loading checkpoint '{}'".format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
else:
    raise Exception("=> No checkpoint found at '{}'".format(ckpt_path))

#####result_path######
ckpt = './result_0' # the directory of checkpoints.
if not os.path.exists(ckpt):
    os.mkdir(ckpt)


model.eval()

tf =TestDataset()
test_loader = DataLoader(tf, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)

miou = 0
mMse=0
for i, sample in enumerate(test_loader):
    image = sample['images'].float().cuda()
    label = sample['labels'].float().cuda()

    with torch.no_grad():
        #print(image.shape)
        image = Variable(image)
        # The dimension of out should be in the dimension of B,C,H,W,D
        out = model(image)
        out=torch.round(out)
        Mse = loss_fn(out, label)
        #print(out.shape)
        out=out.cpu().numpy().squeeze()
        label=label.cpu().numpy().squeeze()
        ####save result
        Image.fromarray(out.astype(np.uint8) * 255).convert('RGB').save(ckpt + '/' + str(i + 1) + '.png')
        ######miou and mse


#         pred_label=out
#         pred_label = np.squeeze(pred_label.reshape(1, -1))
#         #print(pred_label)
#
#         true_label = label
#         true_label = np.squeeze(true_label.reshape(1, -1))
#         confusion_mat = confusion_matrix(true_label, pred_label)
#         print(confusion_mat)
#         iou=confusion_mat[1][1]/(confusion_mat[1][1]+confusion_mat[0][1]+confusion_mat[1][0])
#         miou += iou
#         mMse +=Mse
#
# miou=miou/len(test_loader)
# mMse=mMse/len(test_loader)
# print(miou)
# print(mMse)








