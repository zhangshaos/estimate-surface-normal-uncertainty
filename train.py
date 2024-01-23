import argparse
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

import funcs.utils as utils
from funcs.losses import compute_loss


def train(model, args, device):
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # logging
    writer = SummaryWriter(args.exp_log_dir)

    # dataloader
    if args.dataset_name == 'nyu':
        from data.dataloader_nyu import NyuLoader
        train_loader = NyuLoader(args, 'train').data
        test_loader = NyuLoader(args, 'test').data
    elif args.dataset_name == 'vcc':
        from data.dataloader_vcc import VCC_Loader, VCC_DatasetParams
        params = VCC_DatasetParams()
        params.mode = 'train'
        params.input_height = args.input_height
        params.input_width = args.input_width
        params.batch_size = args.batch_size
        params.num_threads = args.workers
        params.data_augmentation_color = args.data_augmentation_color
        params.data_augmentation_hflip = args.data_augmentation_hflip
        params.data_augmentation_random_crop = args.data_augmentation_random_crop
        params.data_record_file = f'./data_split/data.txt'
        train_loader = VCC_Loader(params).data
        params.mode = 'test'
        test_loader = VCC_Loader(params).data
    else:
        raise Exception('invalid dataset name')

    # define losses
    loss_fn = compute_loss(args)

    # optimizer
    if args.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        params = [{"params": model.get_1x_lr_params(), "lr": args.lr / 10},
                  {"params": model.get_10x_lr_params(), "lr": args.lr}]
    optimizer = optim.AdamW(params, weight_decay=args.weight_decay, lr=args.lr)

    # learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
                                              max_lr=args.lr,
                                              epochs=args.n_epochs,
                                              steps_per_epoch=len(train_loader),
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)

    # cudnn setting
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()

    # start training
    total_iter = 0
    model.train()
    loss_ = 0.
    for epoch in range(args.n_epochs):
        t_loader = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{args.n_epochs}. Loop: Train")
        for data_dict in t_loader:
            optimizer.zero_grad()
            total_iter += args.batch_size

            # data to device
            img = data_dict['img'].to(device)
            gt_norm = data_dict['norm'].to(device)
            gt_norm_mask = data_dict['norm_valid_mask'].to(device)

            # forward pass
            if args.use_baseline:
                norm_out = model(img)
                loss = loss_fn(norm_out, gt_norm, gt_norm_mask)
                norm_out_list = [norm_out]
            else:
                norm_out_list, pred_list, coord_list = model(img, gt_norm_mask=gt_norm_mask, mode='train')
                loss = loss_fn(pred_list, coord_list, gt_norm, gt_norm_mask)

            loss_ = float(loss.data.cpu().numpy())
            t_loader.set_description(f"Epoch: {epoch + 1}/{args.n_epochs}. Loop: Train. Loss: {loss_:.5f}")
            t_loader.refresh()
            writer.add_scalar('train/loss', loss_, global_step=total_iter)

            # back-propagate
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # lr scheduler
            scheduler.step()

            # visualize
            if (total_iter % args.visualize_every) < args.batch_size:
                utils.visualize(args, img, gt_norm, gt_norm_mask, norm_out_list, total_iter)

            # save model
            if (total_iter % args.validate_every) < args.batch_size:
                model.eval()
                target_path = f'{args.exp_model_dir}/checkpoint_iter_{total_iter:010d}_loss_{loss_:.6f}.pt'
                torch.save({"model": model.state_dict(),
                            "iter": total_iter}, target_path)
                print(f'model saved / path: {target_path}')
                # validate(model, args, test_loader, device, total_iter, args.eval_acc_txt)
                model.train()

                # empty cache
                # torch.cuda.empty_cache()

    # save last model
    model.eval()
    target_path = f'{args.exp_model_dir}/checkpoint_iter_{total_iter:010d}_loss_{loss_:.6f}.pt'
    torch.save({"model": model.state_dict(),
                "iter" : total_iter}, target_path)
    print(f'model saved / path: {target_path}')
    # validate(model, args, test_loader, device, total_iter, args.eval_acc_txt)

    # empty cache
    # torch.cuda.empty_cache()

    return model


def validate(model, args, test_loader, device, total_iter, where_to_write, vis_dir=None):
    with torch.no_grad():
        total_normal_errors = None
        for data_dict in tqdm(test_loader, desc="Loop: Validation"):

            # data to device
            img = data_dict['img'].to(device)
            gt_norm = data_dict['norm'].to(device)
            gt_norm_mask = data_dict['norm_valid_mask'].to(device)

            # forward pass
            if args.use_baseline:
                norm_out = model(img)
            else:
                norm_out_list, _, _ = model(img, gt_norm_mask=gt_norm_mask, mode='test')
                norm_out = norm_out_list[-1]

            # upsample if necessary
            if norm_out.size(2) != gt_norm.size(2):
                norm_out = F.interpolate(norm_out, size=[gt_norm.size(2), gt_norm.size(3)], mode='bilinear', align_corners=True)

            pred_norm = norm_out[:, :3, :, :]  # (B, 3, H, W)
            pred_kappa = norm_out[:, 3:, :, :]  # (B, 1, H, W)

            prediction_error = torch.cosine_similarity(pred_norm, gt_norm, dim=1)
            prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
            E = torch.acos(prediction_error) * 180.0 / np.pi

            mask = gt_norm_mask[:, 0, :, :]
            if total_normal_errors is None:
                total_normal_errors = E[mask]
            else:
                # fixme 此处爆显存，total_normal_errors 太长了
                total_normal_errors = torch.cat((total_normal_errors, E[mask]), dim=0)

        total_normal_errors = total_normal_errors.data.cpu().numpy()
        metrics = utils.compute_normal_errors(total_normal_errors)
        utils.log_normal_errors(metrics, where_to_write, first_line='total_iter: {}'.format(total_iter))
        return metrics


if __name__ == '__main__':
    # Arguments ########################################################################################################
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args

    # directory
    parser.add_argument('--exp_dir', default='./experiments', type=str, help='directory to store experiment results')
    parser.add_argument('--exp_name', default='exp00_test', type=str, help='experiment name')
    parser.add_argument('--visible_gpus', default='01', type=str, help='gpu to use')

    # model architecture
    parser.add_argument("--pretrained", default='none', type=str, help="{nyu, scannet, vcc}")
    parser.add_argument('--architecture', default='GN', type=str, help='{BN, GN}')
    parser.add_argument("--use_baseline", action="store_true", help='use baseline encoder-decoder (no pixel-wise MLP, no uncertainty-guided sampling')
    parser.add_argument('--sampling_ratio', default=0.4, type=float)
    parser.add_argument('--importance_ratio', default=0.7, type=float)

    # loss function
    parser.add_argument('--loss_fn', default='UG_NLL_ours', type=str, help='{L1, L2, AL, NLL_vMF, NLL_ours, UG_NLL_vMF, UG_NLL_ours}')

    # training
    parser.add_argument('--n_epochs', default=5, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--validate_every', default=5000, type=int, help='validation period')
    parser.add_argument('--visualize_every', default=1000, type=int, help='visualization period')
    # parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    parser.add_argument("--workers", default=12, type=int, help="Number of workers for data loading")

    # optimizer setup
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
    parser.add_argument('--lr', default=0.000357, type=float, help='max learning rate')
    parser.add_argument('--same_lr', default=False, action="store_true", help="Use same LR for all param groups")
    parser.add_argument('--grad_clip', default=0.1, type=float)
    parser.add_argument('--div_factor', default=25.0, type=float, help="Initial div factor for lr")
    parser.add_argument('--final_div_factor', default=10000.0, type=float, help="final div factor for lr")

    # dataset
    parser.add_argument("--dataset_name", default='vcc', type=str, help="{vcc, nyu}")

    # dataset - preprocessing
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--input_width', default=640, type=int)

    # dataset - augmentation
    parser.add_argument("--data_augmentation_color", default=True, action="store_true")
    parser.add_argument("--data_augmentation_hflip", default=True, action="store_true")
    parser.add_argument("--data_augmentation_random_crop", default=False, action="store_true")

    # read arguments from txt file
    if sys.argv.__len__() == 2 and '.txt' in sys.argv[1]:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.mode = 'train'

    # create experiment directory
    args.exp_dir = f'{args.exp_dir}/{args.exp_name}'
    args.exp_model_dir = f'{args.exp_dir}/models'    # store model checkpoints
    args.exp_vis_dir = f'{args.exp_dir}/vis'         # store training images
    args.exp_log_dir = f'{args.exp_dir}/log'         # store log
    utils.make_dir_from_list([args.exp_dir, args.exp_model_dir, args.exp_vis_dir, args.exp_log_dir])
    print(args.exp_dir)

    utils.save_args(args, f'{args.exp_log_dir}/params.txt')  # save experiment parameters
    args.eval_acc_txt = f'{args.exp_log_dir}/eval_acc.txt'

    # train
    args.gpu = 0

    # define model
    if args.use_baseline:
        from models.baseline import NNET
    else:
        from models.NNET import NNET
    model = NNET(args)
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # load checkpoint
    if args.pretrained != 'none':
        checkpoint = args.pretrained
        print(f'loading checkpoint... {checkpoint}')
        model = utils.load_checkpoint(checkpoint, model)
        print('loading checkpoint... / done')

    train(model, args, device=args.gpu)