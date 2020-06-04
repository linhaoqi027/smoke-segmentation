from DSS import DSS
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset import TrainDataset
import os
import argparse
import math

from torch.utils.data import Dataset
import numpy as np
import random
import os
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import confusion_matrix





ckpt = './saved_models_deform' # the directory of checkpoints.
if not os.path.exists(ckpt):
    os.mkdir(ckpt)

particular_epoch = 0
save_epochs_steps = 1

start_epoch=1
num_epochs = 500
batch_size=4


tf = TrainDataset()
train_loader = DataLoader(tf, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)



model = DSS()
model = torch.nn.DataParallel(model, device_ids=list(range(1))).cuda()
cudnn.benchmark = True


optimizer = optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-5,momentum = 0.9)
criterion = nn.BCELoss()

#ckpt_path = 'saved_models/100_checkpoint.pth.tar'
ckpt_path = 'saved_models_deform/25_checkpoint.pth.tar'

# if os.path.isfile(ckpt_path):
#     print("=> Loading checkpoint '{}'".format(ckpt_path))
#     checkpoint = torch.load(ckpt_path)
#     model_dict = model.state_dict()
#     # 筛除不加载的层结构
#     pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
#     # 更新当前网络的结构字典
#     model_dict.update(pretrained_dict)
#     model.load_state_dict(model_dict)
#
#     print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
# else:
#     raise Exception("=> No checkpoint found at '{}'".format(ckpt_path))
if os.path.isfile(ckpt_path):
    print("=> Loading checkpoint '{}'".format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['opt_dict'])
    start_epoch = checkpoint['epoch']

    print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
else:
    raise Exception("=> No checkpoint found at '{}'".format(ckpt_path))



writer=SummaryWriter()

for epoch in range(start_epoch+1,num_epochs):
    train_loss = 0
    # train_acc = 0
    # train_mean_iu = 0

    prev_time = datetime.now()
    model = model.train()
    for iteration, sample in enumerate(train_loader):
        # image = Variable(data[0].float()).cuda()
        # label = Variable(data[1].long()).cuda()
        image = sample['images'].float()
        target = sample['labels'].float()
        image = Variable(image).cuda()
        label = Variable(target).cuda()
        # print(label.shape)
        # forward
        out = model(image)
        out=torch.squeeze(out)
        # out= out.contiguous().view(-1).cuda()
        # label = label.contiguous().view(-1).cuda()
        loss = criterion(out, label)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(loss.item()*image.size(0))
        train_loss += loss.item()*image.size(0)

        # pred_label = out.data.cpu().numpy()
        # pred_label=np.round(pred_label)
        # pred_label = np.squeeze(pred_label.reshape(1, -1))
        #
        # true_label = label.data.cpu().numpy()
        # true_label = np.squeeze(true_label.reshape(1, -1))
        # confusion_mat = confusion_matrix(true_label, pred_label)
        # class_accuracy=confusion_mat[0][0]/(confusion_mat[0][0]+confusion_mat[0][1])
        # miou=confusion_mat[0][0]/(confusion_mat[0][0]+confusion_mat[0][1]+confusion_mat[1][0])
        #
        # train_acc += class_accuracy
        # train_mean_iu += miou

    # net = net.eval()
    # eval_loss = 0
    # eval_acc = 0
    # eval_mean_iu = 0
    # for data in valid_data:
    #     im = Variable(data[0].cuda(), volatile=True)
    #     labels = Variable(data[1].cuda(), volatile=True)
    #     # forward
    #     out = net(im)
    #     out = F.log_softmax(out, dim=1)
    #     loss = criterion(out, labels)
    #     eval_loss += loss.data[0]
    #
    #     pred_labels = out.max(dim=1)[1].data.cpu().numpy()
    #     pred_labels = [i for i in pred_labels]
    #
    #     true_labels = labels.data.cpu().numpy()
    #     true_labels = [i for i in true_labels]
    #
    #     eval_metrics = eval_semantic_segmentation(pred_labels, true_labels)
    #
    # eval_acc += eval_metrics['mean_class_accuracy']
    # eval_mean_iu += eval_metrics['miou']
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    epoch_str = ('Epoch: {}, Train Loss: {:.5f}'.format(epoch, train_loss / len(train_loader)))
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(epoch_str + time_str )
    writer.add_scalar('loss', train_loss / len(train_loader), epoch)
    # writer.add_scalar('acc', train_acc / len(train_loader), epoch)
    # writer.add_scalar('mean_iou', train_mean_iu / len(train_loader), epoch)



    if epoch > particular_epoch:
        if epoch % save_epochs_steps == 0:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, \
                       ckpt + '/' + str(epoch) + '_checkpoint.pth.tar')

writer.close()