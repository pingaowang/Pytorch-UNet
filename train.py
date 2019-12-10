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

from play import iou, iou_all

from tensorboardX import SummaryWriter
from utils.loss import iou_loss


N_CHANNELS = 3
N_CLASSES_AREA = 8
N_CLASSES_LINE = 4

save_interval = 10
print_interval = 100


def fit(net,
        tf_writer,
        epochs=5,
        batch_size=1,
        lr=0.0001,
        val_percent=0.05,
        save_cp=True,
        gpu=False,
        img_scale=1,
        l2=1e-8,
        mom=0.9,
        n_classes=4,
        loss_function='bce',
        alpha_non_zero = 1
        ):

    # dir_png = "data/caddata_line_v2_1_mini/png"
    # dir_mask = "data/caddata_line_v2_1_mini/mask"
    dir_png = os.path.join(args.data, 'png')
    dir_mask = os.path.join(args.data, 'mask')
    # dir_mask = "data/mini/mask"

    dir_checkpoint = os.path.join(args.log, 'checkpoints')
    if not os.path.isdir(dir_checkpoint):
        os.mkdir(dir_checkpoint)

    ids = get_ids(dir_png)
    ids = split_ids(ids, n=1)
    l_ids = list(ids)
    l_ids_train = l_ids[:-108]
    l_ids_val = l_ids[-108:]

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

    if loss_function == 'bce':
        criterion = nn.BCELoss()
    elif loss_function == 'mse':
        criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()

    with open(os.path.join(args.log, 'log.txt'), 'w+') as f_log:
        for epoch in range(epochs):
            # print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
            # optimizer = optim.SGD(net.parameters(),
            #                       lr=lr * (args.lr_decay ** epoch),
            #                       momentum=mom,
            #                       weight_decay=l2)
            optimizer = optim.Adam(net.parameters(),
                                   lr=lr * (args.lr_decay ** epoch),
                                   weight_decay=l2)
            # optimizer = optim.RMSprop(net.parameters(),
            #                           lr=lr * (args.lr_decay ** epoch),
            #                           weight_decay=l2)
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
            epoch_acc_all = 0

            i_out = 0
            for i, b in enumerate(batch(train, batch_size)):
                imgs = np.array([j[0] for j in b]).astype(np.float32)
                true_masks = np.array([j[1] for j in b])

                imgs = torch.from_numpy(imgs)
                true_masks = torch.from_numpy(true_masks)
                true_masks = np.transpose(true_masks, (0, 3, 1, 2))

                assert imgs.size()[1] == N_CHANNELS
                assert true_masks.size()[1] == n_classes
                assert true_masks.size()[2] == imgs.size()[2]
                assert true_masks.size()[3] == imgs.size()[3]

                if gpu:
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()

                masks_pred = net(imgs)

                # view(-1)
                masks_probs_flat = masks_pred.view(-1)
                if gpu:
                    true_masks_flat = true_masks.view(-1)
                else:
                    true_masks_flat = true_masks.reshape(-1)
                true_masks_flat = true_masks_flat.float()

                # loss
                loss = criterion(masks_probs_flat, true_masks_flat)
                in_nonzero = torch.nonzero(true_masks_flat)
                loss_nonzero = criterion(masks_probs_flat[in_nonzero], true_masks_flat[in_nonzero])
                if in_nonzero.size(0) != 0:
                    loss = loss + alpha_non_zero * loss_nonzero
                epoch_loss += loss.item()

                true_masks_flat_bin = true_masks_flat.unsqueeze(0)
                masks_probs_flat_bin = (masks_probs_flat > 0.5).float().unsqueeze(0)
                this_dice = dice_coeff(masks_probs_flat_bin, true_masks_flat_bin).item()
                epoch_tot += this_dice

                # e = np.array(masks_probs_flat_bin.cpu())
                # f = np.array(true_masks_flat_bin.cpu())
                acc_train = iou(np.array(true_masks_flat_bin.cpu()), np.array(masks_probs_flat_bin.cpu()))
                acc_train_all = iou_all(np.array(true_masks_flat_bin.cpu()), np.array(masks_probs_flat_bin.cpu()))
                epoch_acc += acc_train
                epoch_acc_all += acc_train_all

                if i % print_interval == print_interval - 1:
                    print('{0} / {1} steps. --- loss: {2:.6f}, IoU_train_nz: {3:.4f}, IoU_train_all: {4:.4f}, dice: {5:.4f}'.format(i, int(N_train / batch_size), epoch_loss / (i+1), epoch_acc / (i+1), epoch_acc_all / (i+1), epoch_tot / (i+1)))
                    f_log.write('{0} / {1} steps. --- loss: {2:.6f}, ACC_train: {3:.4f}, IoU_train_all: {4:.4f}, dice: {4:.4f}'.format(i, int(N_train / batch_size), epoch_loss / (i+1), epoch_acc / (i+1), epoch_acc_all / (i+1), epoch_tot / (i+1)) + '\n')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                i_out = i

            print('Epoch {} finished ! Loss: {}, IoU: {}, IoU_all: {}, dice: {}'.format(epoch, epoch_loss / (i_out+1), epoch_acc / (i_out+1), epoch_acc_all / (i_out+1), epoch_tot / (i+1)))
            f_log.write('Epoch finished ! Loss: {}, IoU: {}, IoU_all: {}, dice: {}'.format(epoch_loss / (i_out+1), epoch_acc / (i_out+1), epoch_acc_all / (i_out+1), epoch_tot / (i+1)) + '\n')
            tf_writer.add_scalar('data/train_loss', epoch_loss / (i_out+1), epoch)
            tf_writer.add_scalar('data/train_iou', epoch_acc / (i_out + 1), epoch)
            tf_writer.add_scalar('data/train_iou_all', epoch_acc_all / (i_out + 1), epoch)
            tf_writer.add_scalar('data/train_dice', epoch_tot / (i_out + 1), epoch)

            ## Evaluate
            """Evaluation without the densecrf with the dice coefficient"""
            net.eval()
            tot_val = 0
            epoch_loss_val = 0
            epoch_acc_val = 0
            epoch_acc_val_all = 0
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

                acc_val = iou(np.array(true_masks_flat_bin_val.cpu()), np.array(masks_probs_flat_bin_val.cpu()))
                acc_val_all = iou_all(np.array(true_masks_flat_bin_val.cpu()), np.array(masks_probs_flat_bin_val.cpu()))
                epoch_acc_val += acc_val
                epoch_acc_val_all += acc_val_all

                tot_val += dice_val

                loss_val = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss_val = loss_val / (i_val + 1)
            epoch_dice_val = tot_val / (i_val + 1)
            epoch_acc_val = epoch_acc_val / (i_val + 1)
            epoch_acc_val_all = epoch_acc_val_all / (i_val + 1)

            # val_dice = eval_net(net, val, gpu)
            print('* Val: Loss: {0:.6f}, IoU: {1:.3f}, IoU_all: {2:.3f}, Dice: {3:.3f}'.format(epoch_loss_val, epoch_acc_val, epoch_acc_val_all, epoch_dice_val))
            f_log.write('* Val: Loss: {0:.6f}, IoU: {1:.3f}, IoU_all: {2:.3f}, Dice: {3:.3f}'.format(epoch_loss_val, epoch_acc_val, epoch_acc_val_all, epoch_dice_val) + '\n')
            tf_writer.add_scalar('data/val_loss', epoch_loss_val, epoch)
            tf_writer.add_scalar('data/val_iou', epoch_acc_val, epoch)
            tf_writer.add_scalar('data/val_iou_all', epoch_acc_val_all, epoch)
            tf_writer.add_scalar('data/val_dice', epoch_dice_val, epoch)

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
    parser.add_option('--task-type', type='str', help='line or area')
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
    parser.add_option('--l2', default=0.00000001, type=float, help="SGD's L2 panelty.")
    parser.add_option('--mom', default=0.9, type=float, help="SGD's momentum.")
    parser.add_option('--loss', default='bce', type=str, help='loss function. [bce, mse]')
    parser.add_option('--nz', type='float', default=1, help='coe of non-zero loss.')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    assert args.task_type in ['line', 'area']
    if args.task_type == 'line':
        N_CLASSES = N_CLASSES_LINE
    elif args.task_type == 'area':
        N_CLASSES = N_CLASSES_AREA

    if not os.path.isdir(args.log):
        os.mkdir(args.log)
    FOLDER_LOG = os.path.join(args.log, 'tfboard')
    if not os.path.isdir(FOLDER_LOG):
        os.mkdir(FOLDER_LOG)
    writer = SummaryWriter(logdir=FOLDER_LOG)

    assert args.loss in ['bce', 'mse']

    net = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        fit(net=net,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            gpu=args.gpu,
            val_percent=args.vp,
            img_scale=1,
            l2=args.l2,
            mom=args.mom,
            n_classes=N_CLASSES,
            tf_writer=writer,
            loss_function=args.loss,
            alpha_non_zero=args.nz
            )  # currently img_scale must equal to 1. old: img_scale=args.scale)
        writer.close()
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
