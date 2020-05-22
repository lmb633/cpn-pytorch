import os
import torch
import torch.nn as nn
from models import CPN, ohkm
import numpy as np
from data_gen import MscocoMulti
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.evaluation import AverageMeter


def train(args):
    torch.manual_seed(2)
    np.random.seed(2)
    checkpoint_path = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    epoch_since_improvement = 0

    dataset = MscocoMulti(args, train=True)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = CPN()
    model = torch.nn.DataParallel(model).cuda()
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mon, weight_decay=args.weight_decay)
    criterion1 = torch.nn.MSELoss().cuda()
    criterion2 = torch.nn.MSELoss(reduce=False).cuda()

    log = SummaryWriter(log_dir='data/log', comment='cpn')
    for epoch in range(start_epoch, args.epoch):
        loss = train_once(model, train_loader, optimizer, [criterion1, criterion2], epoch)
        log.add_scalar('cpn_loss', loss, epoch)
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


def train_once(model, train_loader, optimizer, criterion, epoch):
    model.train()
    losses = AverageMeter()
    loss_list = []
    criterion1, criterio2 = criterion
    for i, (inputs, targets, valid, meta) in enumerate(train_loader):
        inputs = inputs.cuda()
        print(inputs.shape)
        refine_target = targets[-1].cuda()
        print(len(targets), print(refine_target.shape))
        global_pred, refine_pred = model(inputs)

        global_loss = 0
        for pred, label in zip(global_pred, targets):
            # label*valid  (mask some heatmap)
            print(pred.shape, label.shape)
            mask = (valid > 1.0).type(torch.FloatTensor).unsqueeze(2).unsqueeze(3)
            print(mask.shape)
            label = label * mask
            print(label.shape)
            global_loss += criterion1(pred, label.cuda()) / 2
        refine_loss = criterio2(refine_pred, refine_target)
        refine_loss = refine_loss.mean(dim=3).mean(dim=2)
        mask = (valid > 0.0).type(torch.FloatTensor)
        print(refine_pred.shape, mask.shape)
        refine_loss = refine_loss * mask
        refine_loss = ohkm(refine_loss, 8)
        loss = global_loss + refine_loss
        losses.update(loss.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss, global_loss, refine_loss)
    return losses.avg


if __name__ == '__main__':
    from torch.autograd import Variable

    x = torch.rand((2, 3, 4, 4))
    print(x.size(0))
    x = x.mean(dim=3).mean(dim=2)
    print(x)
    x = x[0]
    print(x)
    topk_val, topk_idx = torch.topk(x, k=2, dim=0, sorted=True)
    print(topk_idx, topk_val)
    loss = torch.gather(x, 0, topk_idx)
    print(loss)