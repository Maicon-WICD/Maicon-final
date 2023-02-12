import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import random
from CustomDataset import *
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from losses import *
from networks import *

import torch.cuda.amp as amp
import GPUtil

parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=20, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="/workspace/datasets/PReNet/logs/PreNet_test_cut_pnsr/", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=200,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="/workspace/datasets/num_Train/",help='path to training datasets')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    
SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(SEED)


def main():

    print('Loading dataset ...\n')
    dataset_train = CustomDataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=2, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    scaler = amp.GradScaler()
    # Build model
    model = PReNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)

    # loss function
    # criterion = nn.MSELoss(size_average=False)
    criterion = PSNRLoss()

    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates
    initial_epoch = 0
    # load the lastest model
    if initial_epoch > 0:
        print('resuming by loading epoch ')
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_iter1601.pth' )))

    # start training
    iter = 0
    max = 0
    for epoch in range(initial_epoch, opt.epochs):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        model.train()
        ## epoch training start
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            iter +=1
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()
            with amp.autocast():
                out_train, _ = model(input_train)
                pixel_metric = criterion(target_train, out_train)
                loss = pixel_metric
            
            scaler.scale(loss).backward()
            
            scaler.step(optimizer)
            scaler.update()
            
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))
            curr = psnr_train
            
            if curr > max:
                max = curr
                torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_iter%d.pth' % (max)))
    # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))

if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrainH') != -1:
            print(opt.data_path.find('RainTrainH'))
            prepare_data_RainTrainH(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('RainTrainL') != -1:
            prepare_data_RainTrainL(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('Rain12600') != -1:
            prepare_data_Rain12600(data_path=opt.data_path, patch_size=100, stride=100)
        else:
            print('unkown datasets: please define prepare datasets function in DerainDataset.py')


    main()
