# Adapted from Torch's MNIST example (main.py) file and SMP's example on Jupyter Notebook

from __future__ import print_function
import os
import argparse
import torch
import time
import tkinter as tk
from tkinter import filedialog
import numpy as np
import torch.optim as optim
import segmentation_models_pytorch as smp
import PP_segmentation_utils as pp
from PP_Datasets import MRIDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Preston\'s MRI-segmentation Arguments')
    parser.add_argument('--mode', type=str, default="train", metavar='M',  # need to wrap the argument in double quotation
                        help='Select either train or test (default: "train")')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to checkpoint")
    parser.add_argument('--num_files', type=int, default=4000, metavar='N',
                        help='Number of files in training; split using tv_ratio (default: 4000)'),
    parser.add_argument('--tv_ratio', type=list, default=[0.8, 0.2],
                        help='Train-validate ratio in a list, must add up to 1, (default: [0.8, 0.2])')
    parser.add_argument('--train_batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--valid_batch_size', type=int, default=4, metavar='N',
                        help='input batch size for validating (default: 4)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=3e-3, metavar='LR',
                        help='learning rate (default: 3e-3)')
    parser.add_argument('--sch_step', type=int, default=10, metavar='sch_step',
                        help='Scheduler step size (default: 10)')
    parser.add_argument('--loss', type=str, default="dl", metavar='LOSS',
                        help='Type of loss - dl or gdl (default: dl)')
    parser.add_argument('--beta', type=float, default=1, metavar='B',
                        help='Beta in dice loss (default: 1)')
    parser.add_argument('--pw', type=float, default=1000, metavar='PW',
                        help='Positive weight in the bce function')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,  # if I specify "--no-cuda," I don't need an argument and this assumes that I am using my CPU. Otherwise, don't specify anything and use GPU by default
                        help='disables CUDA training')
    parser.add_argument('--just_show', type=int, default=None,
                        help='set this flag when you want to skip the test metrics')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    start_time = time.time()
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # Model Definition
    ENCODER = 'se_resnext50_32x4d'
    # ENCODER = 'vgg16'
    ENCODER_WEIGHTS = 'imagenet'  # None or 'imagenet' (if None, then weights are randomly initialized)
    CLASSES = ['tumor']  # only 1 class in this example
    ACTIVATION = 'sigmoid'
    # could be None for logits or 'softmax2d' for multicalss segmentation; sigmoid for binary classification
    DEVICE = "cuda" if use_cuda else "cpu"

    # create segmentation model
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),  # only 1 class in this case
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, pretrained='imagenet')

    if args.loss == "dl":
        loss = smp.utils.losses.DiceLoss(beta=args.beta)
    elif args.loss == "gdl":
        loss = smp.utils.losses.GeneralizedDiceLoss()
    elif args.loss == "dl+bce":
        pw = torch.FloatTensor([args.pw])  # gonna use 1000 here because we of the pos_thresh we set in this pruned dataset
        pw = torch.reshape(pw, (1, 1, 1, 1))
        loss = smp.utils.losses.DiceLoss(beta=args.beta) + smp.utils.losses.BCEWithLogitsLoss(pos_weight=pw)
    elif args.loss == "dl+log(bce)":  # doesn't work yet
        pw = torch.FloatTensor([args.pw])
        pw = torch.reshape(pw, (1, 1, 1, 1))
        loss = smp.utils.losses.DiceLoss(beta=args.beta) + torch.log(smp.utils.losses.BCEWithLogitsLoss(pos_weight=pw))
    else:
        raise ValueError("Loss can only be dl or gdl for now")

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    data_dir = 'D:\\MRI Segmentation\\data'
    model_dir = r"C:\Users\prestonpan\PycharmProjects\Segmentation_example\runs"

    if args.mode == "train":
        comment = f'lr={args.lr}, loss={args.loss}, epochs={args.epochs}, ' \
                  f'num_files={args.num_files}, sch_step={args.sch_step}, train_batch_size={args.train_batch_size}, ' \
                  f'beta={args.beta}, pw={args.pw}'
        tb = SummaryWriter(comment=comment)
        print(comment)  # to verify that the hyperparameter values are set correctly

        dfTrain, dfVal = pp.prepare_csv(data_dir, args.tv_ratio, num_files=args.num_files, mode=args.mode)

        # Gonna modify the loader so that it looks more similar to the Dataset class used in smp's example
        train_dataset = MRIDataset(dfTrain,  # might just remove this kwarg later on
                                   classes=CLASSES,
                                   augmentation=pp.get_training_augmentation(),
                                   preprocessing=pp.get_preprocessing(preprocessing_fn))
        valid_dataset = MRIDataset(dfVal,
                                   classes=CLASSES,
                                   augmentation=pp.get_training_augmentation(),
                                   preprocessing=pp.get_preprocessing(preprocessing_fn))

        train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=12)
        valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=4)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.90, weight_decay=1e-6, nesterov=True)
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)  # this works very poorly

        start_epoch = 0
        max_score = 0

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.sch_step,
                                              gamma=args.gamma)  # every step_size number of epoch, the lr is multiplied by args.gamma - reduces learning rate due to subgradient method

        # Loads the checkpoint and update some parameters
        if args.checkpoint is not None:
            root = tk.Tk()
            checkpoint_path = filedialog.askopenfilename(parent=root, initialdir=model_dir,
                                                         title='Please select a model (.pth)')
            model, optimizer, scheduler, max_score, start_epoch = \
                pp.load_checkpoint(model, optimizer, scheduler, filename=checkpoint_path)

        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=DEVICE,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
            verbose=True,
        )

        stagnant_epoch = 0
        for i in range(start_epoch, args.epochs):
            print('\nEpoch: {}'.format(i+1))  # 1-index the epoch number
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            tb.add_scalar('Loss', train_logs[loss.__name__], i)
            tb.add_scalar('IOU', train_logs['iou_score'], i)

            scheduler.step()

            if max_score < valid_logs['iou_score']:  # we really care more about the validation metric because
                stagnant_epoch = 0
                # training metric can be overfit, which is unrepresentative
                max_score = valid_logs['iou_score']
                print('New best IOU: %.5f' % max_score)
                print('Saving checkpoint...')
                state = {'model': model, 'epoch': args.epochs, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                         'iou_score': max_score}
                torch.save(state, './best MRI models/{}.pth'.format(comment))
            else:
                stagnant_epoch = stagnant_epoch + 1
                if stagnant_epoch > 8:
                    print('max iou remained stagnant for the past 8 epochs, returning early')
                    return

        total_time = time.time() - start_time
        print('Training took %.2f seconds' % total_time)


    elif args.mode == "test":

        dfTest = pp.prepare_csv(data_dir, args.tv_ratio, num_files=-1, mode=args.mode)  # number of testing images is fixed

        root = tk.Tk()
        model_path = filedialog.askopenfilename(parent=root, initialdir=model_dir,
                                                title='Please select a model (.pth)')
        root.destroy()

        # best_model = torch.load(model_path) # This line doesn't work anymore because the model is now saved in a dict
        best_model_dict = torch.load(model_path)
        best_model = best_model_dict['model']
        print('model loaded!')

        test_dataset = MRIDataset(dfTest, classes=CLASSES,
                                  preprocessing=pp.get_preprocessing(preprocessing_fn))

        # We have pre-processing here (to make sure the prediction works properly), but no need augmentation
        # because we're not training anymore
        test_dataloader = DataLoader(test_dataset)
        test_epoch = smp.utils.train.ValidEpoch(
            model=best_model,
            loss=loss,
            metrics=metrics,
            device=DEVICE,
        )

        if args.just_show is not None:
            num_to_display = args.just_show
        else:
            logs = test_epoch.run(test_dataloader)
            num_to_display = 5

        # # dataset for visualization without augmentation
        test_dataset_vis = MRIDataset(dfTest, classes=CLASSES)  # no preprocessing; this is the native image

        for i in range(num_to_display):
            n = np.random.choice(len(test_dataset_vis))

            image_vis = test_dataset_vis[n][0].astype('int16')  # arranged in H * W * 3 (3 = RGB channels)
            image, gt_masks = test_dataset[n]
            # image has 3 channels (RGB) -> shape(image) = 3 * H * W, different from image_vis because of preprocessing
            # gt_masks has C classes -> shape(gt_mask) = C * H * W (C = Classes)

            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)  # shape = 1 * H * W * 3 (3 = RGB channels)
            pr_masks = best_model.predict(x_tensor)  # shape = 1 * H * W * C (C = number of classes)
            pr_masks = (pr_masks.squeeze(0).cpu().numpy().round())  # shape = C * H * W  (squeeze the 0th dimensions)

            # Index through the classes and look at them one at a time
            for j in range(len(CLASSES)):
                gt_mask = gt_masks[j].squeeze()
                pr_mask = pr_masks[j].squeeze()


                pp.visualize_2(
                    image=image_vis,
                    gt=gt_mask,
                    pr=pr_mask,
                    iou=1 - loss.forward(torch.from_numpy(gt_mask), torch.from_numpy(pr_mask))
                )


if __name__ == '__main__':
    main()