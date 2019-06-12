import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from dice_loss import dice_coeff
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch


N_CHANNELS = 1
N_CLASSES = 2

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=1):

    dir_png = "data/our_dataset/png"
    dir_mask = "data/our_dataset/mask"
    dir_checkpoint = 'checkpoints/'
    if not os.path.isdir(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    ids = get_ids(dir_png)
    ids = split_ids(ids, n=1)

    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        optimizer = optim.SGD(net.parameters(),
                              lr=lr * (args.lr_decay ** epoch),
                              momentum=0.9,
                              weight_decay=0.0005)
        print("Current lr = {}".format(lr * (args.lr_decay ** epoch)))
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_png, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_png, dir_mask, img_scale)

        epoch_loss = 0
        epoch_tot = 0

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([j[0] for j in b]).astype(np.float32)
            true_masks = np.array([j[1] for j in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)
            true_masks_flat = true_masks_flat.float()

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            true_masks_flat_bin = true_masks_flat.unsqueeze(0)
            masks_probs_flat_bin = (masks_probs_flat > 0.5).float().unsqueeze(0)
            epoch_tot += dice_coeff(masks_probs_flat_bin, true_masks_flat_bin).item()



            if i % 500 == 0:
                print('{0} / {1} steps. --- loss: {2:.6f}, Dice: {3:.4f}'.format(i, N_train, epoch_loss / (i+1), epoch_tot / (i+1)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}, Dice: {}'.format(epoch_loss / (i+1), epoch_tot / (i+1)))

        if 1:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-v', '--val_percent', dest='vp', type='float',
                      default=0.05, help='percent val set of all')
    parser.add_option('-d', '--lr_decay', dest='lr_decay', type='float',
                      default=0.1, help='learning rate decay per epoch')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  val_percent = args.vp,
                  img_scale=1)  # currently img_scale must equal to 1. old: img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
