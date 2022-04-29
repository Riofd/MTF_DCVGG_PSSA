import numpy as np
import os
from img_dataset import DimDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import torchvision
import torch
import torch.nn as nn
import argparse
import scipy.io as sio
from model import VGG_8
import time

torch.cuda.set_device(1)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--weight_decay', type=float, default=0.0005,
                    help='weight decay')
parser.add_argument('--epochs', type=int, default=100,
                    help='maximum number of epochs')
args = parser.parse_args()


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img_dir = {'NEPG': 'E:/Users/zrf/转移文件夹/样本/image_NCPG/6_20',
           'CEPRI': 'E:/Users/zrf/转移文件夹/样本/知识',
           'CEPRI_K_means': 'D:/Users/zrf/TSC/data/CEPRI/K_Means',
           'CEPRI_N_1': 'D:/Users/zrf/TSC/data/CEPRI/N_1'}
dataset = DimDataset(root=img_dir['CEPRI'], rotor_gaf=False, vol_gaf=False, transform=transform)

batch_size = 64
shuffle_dataset = True
random_seed = 35

# Creating data indices for training and validation splits:
if shuffle_dataset:
    np.random.seed(random_seed)
dataset_size = len(dataset)
indices = np.arange(dataset_size)
indices = np.random.permutation(indices)
train_indices = indices[:int(0.7*dataset_size)]
val_indices = indices[int(0.7*dataset_size):int(0.85*dataset_size)]
test_indices = indices[int(0.85*dataset_size):]
# train_indices = indices[:2600]
# val_indices = indices[2600:5200]
# test_indices = indices[2600:5200]
val_size = len(val_indices)
train_indices = train_indices[:2000]
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)
print('Data loading...')
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
print('Done!')
torch.set_num_threads(1)
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VGG_8().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            momentum=args.momentum)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)


def train():
    # if epoch == 10:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.001
    # if epoch == 50:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.0005
    # if epoch == 100:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.0002
    model.train()
    loss_all = 0
    best_loss = 100
    print('Epoch:{}'.format(epoch))
    for (rotor, vol, data_y) in train_loader:
        rotor = rotor.to(device)
        vol = vol.to(device)
        data_y = data_y.to(device)
        data_y = data_y.long()
        optimizer.zero_grad()
        out = model(rotor, vol)
        train_loss = criterion(out, data_y)
        loss_all += train_loss.item()
        train_loss.backward()
        optimizer.step()
        if train_loss.item() < best_loss:
            best_loss = train_loss.item()
    train_avg_loss = loss_all / len(train_loader)
    print("Training average loss: {:.4f}, Training best loss: {:.4f}".format(train_avg_loss, best_loss))
    return train_avg_loss, best_loss


def test(loader):
    model.eval()
    correct = 0.
    loss = 0.
    topk = 0.
    with torch.no_grad():
        for (rotor, vol, data_y) in loader:
            rotor = rotor.to(device)
            vol = vol.to(device)
            data_y = data_y.to(device)
            data_y = data_y.long()
            test_labels = data_y
            out = model(rotor, vol)
            pred = out.max(1)[1]
            _, maxk = torch.topk(out, k=2, dim=-1)
            test_labels = test_labels.view(-1, 1)
            topk += (test_labels == maxk).sum().item()
            correct += pred.eq(data_y).sum().item()
            loss += criterion(out, data_y).item()
    return correct / val_size, topk / val_size, loss / len(loader)


loss_list = []
acc_list = []
top2_acc_list = []
train_loss = []
best_loss = []
best_acc = 0.
lowest_loss = 100.
OUT_PATH = './checkpoint/'
LOAD_PATH = './checkpoint/MTF_NCPG_epoch200/'
model.load_state_dict(torch.load(LOAD_PATH+'checkpoint-best-loss.pkl'))
# frozen_layers = ['fc1', 'bn1', 'fc3', 'conv3_128b_vol', 'conv3_128b_rotor']
frozen_layers = ['fc1', 'bn1', 'fc3']
for name, param in model.named_parameters():
    if name[:-7] in frozen_layers or name[:-5] in frozen_layers or name[:-9] in frozen_layers:
        param.requires_grad = True
    else:
        param.requires_grad = False
for epoch in range(1, args.epochs+1):
    start_time = time.time()
    training_loss, b_loss = train()
    train_loss.append(training_loss)
    best_loss.append(b_loss)
    val_acc, top2_acc, val_loss = test(val_loader)
    lr = optimizer.param_groups[0]['lr']
    # lr = scheduler.optimizer.param_groups[0]['lr']
    # scheduler.step()
    loss_list.append(val_loss)
    acc_list.append(100. * val_acc)
    top2_acc_list.append(100. * top2_acc)
    end_time = time.time()
    if best_acc < val_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), OUT_PATH+'checkpoint-best-acc.pkl')
    if lowest_loss > val_loss:
        lowest_loss = val_loss
        torch.save(model.state_dict(), OUT_PATH+'checkpoint-best-loss.pkl')
    print("Validation loss:{:.4f}\taccuracy:{:.4f}\ttop2_acc:{:.4f}\tlr:{:.3e}\tTime:{:.2f}"
          .format(val_loss, val_acc, top2_acc, lr, end_time-start_time))
print("best acc:{:.4f}\tlowest_loss:{:.4f}".format(best_acc, lowest_loss))
train_info = {'val_loss': loss_list, 'val_acc': acc_list, 'top2_acc': top2_acc_list,
              'train_loss': train_loss, 'best_loss': best_loss}
sio.savemat('./result/train_info.mat', mdict=train_info)

model.load_state_dict(torch.load(OUT_PATH+'checkpoint-best-loss.pkl'))
test_acc, top2_acc, test_loss = test(test_loader)
print('Test: {:.4f}, Top2_Acc: {:.4f}'.format(test_acc, top2_acc))

