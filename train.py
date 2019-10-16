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
import random


N_CHANNELS = 3
N_CLASSES = 8

save_interval = 10
print_interval = 100


def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=1):

    # dir_png = "data/caddata_line_v2_1_mini/png"
    # dir_mask = "data/caddata_line_v2_1_mini/mask"
    dir_png = os.path.join(args.data, 'png')
    dir_mask = os.path.join(args.data, 'mask')
    # dir_mask = "data/mini/mask"
    if not os.path.isdir(args.log):
        os.mkdir(args.log)

    dir_checkpoint = os.path.join(args.log, 'checkpoints')
    if not os.path.isdir(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    ids = get_ids(dir_png)
    ids = split_ids(ids, n=1)
    l_ids = list(ids)
    l_ids_train = l_ids[:-30]
    l_ids_val = l_ids[-30:]

    # iddataset = split_train_val(ids, val_percent)
    iddataset = {
        'train': l_ids_train,
        'val': l_ids_val
         }

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

    with open(os.path.join(args.log, 'log.txt'), 'w+') as f_log:
        for epoch in range(epochs):
            # print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
            optimizer = optim.SGD(net.parameters(),
                                  lr=lr * (args.lr_decay ** epoch),
                                  momentum=0.9,
                                  weight_decay=0.0005)
            print("Current lr = {}".format(lr * (args.lr_decay ** epoch)))
            f_log.write("Current lr = {}".format(lr * (args.lr_decay ** epoch)) + '\n')

            net.train()

            # shuffle training set
            random.shuffle(l_ids_train)
            iddataset = {
                'train': l_ids_train,
                'val': l_ids_val
            }
            # reset the generators
            train = get_imgs_and_masks(iddataset['train'], dir_png, dir_mask, img_scale)
            val = get_imgs_and_masks(iddataset['val'], dir_png, dir_mask, img_scale)

            epoch_loss = 0
            epoch_tot = 0
            epoch_acc = 0

            i_out = 0
            for i, b in enumerate(batch(train, batch_size)):
                imgs = np.array([j[0] for j in b]).astype(np.float32)
                true_masks = np.array([j[1] for j in b])

                imgs = torch.from_numpy(imgs)
                true_masks = torch.from_numpy(true_masks)
                true_masks = np.transpose(true_masks, (0, 3, 1, 2))

                assert imgs.size()[1] == N_CHANNELS
                assert true_masks.size()[1] == N_CLASSES
                assert true_masks.size()[2] == imgs.size()[2]
                assert true_masks.size()[3] == imgs.size()[3]

                if gpu:
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()

                masks_pred = net(imgs)
                masks_probs_flat = masks_pred.view(-1)

                if gpu:
                    true_masks_flat = true_masks.view(-1)
                else:
                    true_masks_flat = true_masks.reshape(-1)
                true_masks_flat = true_masks_flat.float()

                loss = criterion(masks_probs_flat, true_masks_flat)
                epoch_loss += loss.item()

                true_masks_flat_bin = true_masks_flat.unsqueeze(0)
                masks_probs_flat_bin = (masks_probs_flat > 0.5).float().unsqueeze(0)
                this_dice = dice_coeff(masks_probs_flat_bin, true_masks_flat_bin).item()
                epoch_tot += this_dice

                e = np.array(masks_probs_flat_bin.cpu())
                f = np.array(true_masks_flat_bin.cpu())
                acc_train = np.mean(e == f)
                epoch_acc += acc_train

                if i % print_interval == print_interval - 1:
                    print('{0} / {1} steps. --- loss: {2:.6f}, ACC_train: {3:.4f}, dice: {4:.4f}'.format(i, int(N_train / batch_size), epoch_loss / (i+1), epoch_acc / (i+1), epoch_tot / (i+1)))
                    f_log.write('{0} / {1} steps. --- loss: {2:.6f}, ACC_train: {3:.4f}, dice: {4:.4f}'.format(i, int(N_train / batch_size), epoch_loss / (i+1), epoch_acc / (i+1), epoch_tot / (i+1)) + '\n')

                    # print('{0} / {1} steps. --- loss: {2:.6f}, Dice: {3:.4f}'.format(i, N_train, epoch_loss / (i + 1),
                    #                                                                  epoch_tot/ (i + 1)))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                i_out = i


            print('Epoch {} finished ! Loss: {}, ACC: {}, dice: {}'.format(epoch, epoch_loss / (i_out+1), epoch_acc / (i_out+1), epoch_tot / (i+1)))
            f_log.write('Epoch finished ! Loss: {}, ACC: {}, dice: {}'.format(epoch_loss / (i_out+1), epoch_acc / (i_out+1), epoch_tot / (i+1)) + '\n')

            ## Evaluate
            """Evaluation without the densecrf with the dice coefficient"""
            net.eval()
            tot_val = 0
            epoch_acc_val = 0
            for i_val, b_val in enumerate(batch(val, batch_size)):
                imgs_val = np.array([j[0] for j in b_val]).astype(np.float32)
                true_masks_val = np.array([j[1] for j in b_val])

                imgs_val = torch.from_numpy(imgs_val)
                true_masks_val = torch.from_numpy(true_masks_val)
                true_masks_val = np.transpose(true_masks_val, (0, 3, 1, 2))

                if gpu:
                    imgs_val = imgs_val.cuda()
                    true_masks_val = true_masks_val.cuda()

                masks_pred_val = net(imgs_val)
                masks_probs_flat_val = masks_pred_val.view(-1)

                if gpu:
                    true_masks_flat_val = true_masks_val.view(-1)
                else:
                    true_masks_flat_val = true_masks_val.reshape(-1)
                true_masks_flat_val = true_masks_flat_val.float()

                true_masks_flat_bin_val = true_masks_flat_val.unsqueeze(0)
                masks_probs_flat_bin_val = (masks_probs_flat_val > 0.5).float().unsqueeze(0)
                dice_val = dice_coeff(masks_probs_flat_bin_val, true_masks_flat_bin_val).item()

                #############
                # masks_pred_val_2 = net(imgs)
                # masks_probs_flat_val_2 = masks_pred_val_2.view(-1)
                # masks_probs_flat_bin_val_2 = (masks_probs_flat_val_2 > 0.5).float().unsqueeze(0)

                # a = np.array(true_masks_flat_bin.cpu())
                b = np.array(true_masks_flat_bin_val.cpu())
                #
                c = np.array(masks_probs_flat_bin_val.cpu())
                # d = np.array(masks_probs_flat_bin_val_2.cpu())

                # e = np.array(masks_probs_flat_bin.cpu())
                # f = np.array(true_masks_flat_bin.cpu())
                # masks_probs_flat_bin, true_masks_flat_bin


                acc_val = np.mean(c == b)
                epoch_acc_val += acc_val
                # acc_train = np.mean(e == f)
                ##############

                tot_val += dice_val
                # print("dice_val:{}".format(dice_val))
            val_dice = tot_val / (i_val + 1)
            epoch_acc_val = epoch_acc_val / (i_val + 1)

            # val_dice = eval_net(net, val, gpu)
            print('Validation ACC: {}'.format(epoch_acc_val) + '\n')
            f_log.write('Validation ACC: {}'.format(epoch_acc_val))

            if save_cp and (epoch % save_interval == save_interval - 1):
                torch.save(net.state_dict(),
                           dir_checkpoint + '/CP{}.pth'.format(epoch + 1))
                print('Checkpoint {} saved !'.format(epoch + 1))
                f_log.write('Checkpoint {} saved !'.format(epoch + 1) + '\n')




def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('--data', type='str', help='folder of dataset.')
    parser.add_option('--log', type='str', help='folder of log.')
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
                  val_percent=args.vp,
                  img_scale=1)  # currently img_scale must equal to 1. old: img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
