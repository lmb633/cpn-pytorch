import os
import torch
import torch.nn as nn
from models import CPN, ohkm
import numpy as np
from data_gen import MscocoMulti
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.evaluation import AverageMeter
from config import cfg
from utils.misc import adjust_learning_rate


def train():
    torch.manual_seed(2)
    np.random.seed(2)
    checkpoint_path = cfg.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    epoch_since_improvement = 0

    dataset = MscocoMulti(cfg, train=True)
    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    model = CPN()
    model = torch.nn.DataParallel(model).cuda()
    if os.path.exists(checkpoint_path):
        print('=========load checkpoint========')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.mon, weight_decay=cfg.weight_decay)
    criterion1 = torch.nn.MSELoss().cuda()
    criterion2 = torch.nn.MSELoss(reduction='none').cuda()

    log = SummaryWriter(log_dir='data/log', comment='cpn')
    for epoch in range(start_epoch, cfg.epoch):
        adjust_learning_rate(optimizer, epoch, cfg.lr_dec_epoch, cfg.lr_gamma)
        loss = train_once(model, train_loader, optimizer, [criterion1, criterion2], epoch, log)
        log.add_scalar('cpn_loss', loss, epoch)
        log.flush()
        if loss < best_loss:
            best_loss = loss
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optm': optimizer,
            }, checkpoint_path)
            epoch_since_improvement = 0
        else:
            epoch_since_improvement += 1
    log.close()


def train_once(model, train_loader, optimizer, criterion, epoch, log):
    model.train()
    losses = AverageMeter()
    criterion1, criterio2 = criterion
    for i, (inputs, targets, valid, meta) in enumerate(train_loader):
        inputs = inputs.cuda()
        # print(inputs.shape)
        # print(len(targets), print(targets[-1].shape))
        refine_target = targets[-1].cuda()
        global_pred, refine_pred = model(inputs)

        global_loss = 0
        for pred, label in zip(global_pred, targets):
            # label*valid  (mask some heatmap)
            # print(pred.shape, label.shape)
            mask = (valid > 1.0).type(torch.FloatTensor).unsqueeze(2).unsqueeze(3)
            # print(mask.shape)
            label = label * mask
            # print(label.shape)
            global_loss += criterion1(pred, label.cuda()) / 2
        refine_loss = criterio2(refine_pred, refine_target)
        refine_loss = refine_loss.mean(dim=3).mean(dim=2)
        mask = (valid > 0.0).type(torch.FloatTensor)
        # print(refine_pred.shape, mask.shape)
        refine_loss = refine_loss * mask.cuda()
        refine_loss = ohkm(refine_loss, 8)
        loss = global_loss + refine_loss
        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        log.add_scalar('loss_epoch_{0}'.format(epoch), loss.item(), i)
        log.flush()
        if i % cfg.print_freq == 0:
            print('epoch: ', epoch, '{0}/{1} loss_avg: {2} global_loss: {3} refine_loss: {4} loss: {5}'.format(i, len(train_loader), losses.avg, global_loss, refine_loss, loss))
    return losses.avg


if __name__ == '__main__':
    train()
