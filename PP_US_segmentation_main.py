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
from PP_Datasets import USDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Preston\'s MRI-segmentation Arguments')
    parser.add_argument('--mode', type=str, default="train", metavar='M',  # need to wrap the argument in double quotation
                        help='Select either train or test (default: "train")')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to checkpoint")
    parser.add_argument('--cropped_input', action='store_true', default=False,  # NOT WORKING?
                        help='Uses cropped data as input to exclude more background')
    parser.add_argument('--num_files', type=int, default=14, metavar='N',
                        help='Number of files in training; split using tv_ratio (default: 14)'),
    parser.add_argument('--tv_ratio', type=list, default=[0.8, 0.2],
                        help='Train-validate ratio in a list, must add up to 1')
    parser.add_argument('--train_batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--valid_batch_size', type=int, default=1, metavar='N',
                        help='input batch size for validating (default: 1)')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=3e-3, metavar='LR',
                        help='learning rate (default: 3e-3)')
    parser.add_argument('--sch_step', type=int, default=5, metavar='sch_step',
                        help='Scheduler step size (default: 5)')
    parser.add_argument('--loss', type=str, default="dl", metavar='LOSS',
                        help='Type of loss - dl or gdl (default: dl)')
    parser.add_argument('--beta', type=float, default=1, metavar='B',
                        help='Beta in dice loss (default: 1)')
    parser.add_argument('--pw', type=float, default=1000, metavar='PW',
                        help='Positive weight in the bce function (default: 1000')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.9)')
    parser.add_argument('--classes', nargs='+', required=False,
                        help='Classes to segment; choose from '
                             '[ring-down, gel-cover, free gel, soft tissue, muscle, bone, root, crown, suture, lip]')
    parser.add_argument('--not_pretrained', action='store_true', default=False,
                       help='Disables imagenet pretrain')
    parser.add_argument('--no_cuda', action='store_true', default=False,  # if I specify "--no-cuda," I don't need an argument and this assumes that I am using my CPU. Otherwise, don't specify anything and use GPU by default
                        help='disables CUDA training')
    parser.add_argument('--just_show', type=int, default=None,
                        help='set this flag when you want to skip the test metrics')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    start_time = time.time()
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    DEVICE = "cuda" if use_cuda else "cpu"



    metrics = [
        smp.utils.metrics.IoU(threshold=None),
    ]

    data_dir = r"G:\My Drive\Umich Research\Dental Segmentation\Data"
    model_dir = r"C:\Users\prestonpan\PycharmProjects\Segmentation_example\runs"

    # Define encoder and preprocessing function to be kept constant between train and test
    ENCODER = 'vgg16'  # maybe we stick with this for now?
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, pretrained='imagenet')

    if args.mode == "train":

        # Model definition
        ENCODER_WEIGHTS = 'imagenet' if not args.not_pretrained else None
        # CLASSES = ['ring-down', 'gel-cover', 'free gel', 'soft tissue',
        #            'muscle', 'bone', 'root', 'crown', 'suture', 'lip']  # don't train the background class
        CLASSES = args.classes
        ACTIVATION = 'softmax2d' if len(CLASSES) > 1 else 'sigmoid'
        print(ACTIVATION)

        # create segmentation model
        model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            classes=len(CLASSES),  # only 1 class in this case
            activation=ACTIVATION,
        )

        if args.loss == "dl":
            loss = smp.utils.losses.DiceLoss(beta=args.beta)
        elif args.loss == "gdl":
            loss = smp.utils.losses.GeneralizedDiceLoss()
        elif args.loss == "dl+bce":
            pw = torch.FloatTensor(
                [args.pw])  # gonna use 1000 here because we of the pos_thresh we set in this pruned dataset
            pw = torch.reshape(pw, (1, 1, 1, 1))
            loss = smp.utils.losses.DiceLoss(beta=args.beta) + smp.utils.losses.BCEWithLogitsLoss(pos_weight=pw)
        elif args.loss == "dl+log(bce)":  # doesn't work yet
            pw = torch.FloatTensor([args.pw])
            pw = torch.reshape(pw, (1, 1, 1, 1))
            loss = smp.utils.losses.DiceLoss(beta=args.beta) + torch.log(
                smp.utils.losses.BCEWithLogitsLoss(pos_weight=pw))
        else:
            raise ValueError("Loss can only be dl or gdl for now")

        comment = f'lr={args.lr}, loss={args.loss}, epochs={args.epochs}, ' \
                  f'num_files={args.num_files}, sch_step={args.sch_step}, train_batch_size={args.train_batch_size}, ' \
                  f'beta={args.beta}, classes={CLASSES}'
        tb = SummaryWriter(comment=comment)
        print(comment)  # to verify that the hyperparameter values are set correctly

        dfTrain, dfVal = pp.prepare_US_csv(data_dir, args.tv_ratio, cropped=args.cropped_input, num_files=args.num_files, mode=args.mode)

        # Gonna modify the loader so that it looks more similar to the Dataset class used in smp's example
        train_dataset = USDataset(dfTrain,  # might just remove this kwarg later on
                                  classes=CLASSES,
                                  augmentation=pp.get_US_training_augmentation(cropped=args.cropped_input),
                                  preprocessing=pp.get_preprocessing(preprocessing_fn))
        valid_dataset = USDataset(dfVal,
                                  classes=CLASSES,
                                  augmentation=pp.get_US_validation_augmentation(),
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
                state = {'model': model, 'epoch': args.epochs, 'loss': loss, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                         'iou_score': max_score, 'classes': args.classes}
                torch.save(state, './best US models/{}.pth'.format(comment))
            else:
                stagnant_epoch = stagnant_epoch + 1
                if stagnant_epoch > 6:
                    print('max iou remained at %.2f for the past 6 epochs, returning early' % max_score)
                    return

        total_time = time.time() - start_time
        print('Training took %.2f seconds' % total_time)


    elif args.mode == "test":

        dfTest = pp.prepare_US_csv(data_dir, args.tv_ratio, cropped=args.cropped_input, num_files=-1, mode=args.mode)  # number of testing images is fixed

        root = tk.Tk()
        model_path = filedialog.askopenfilename(parent=root, initialdir=model_dir,
                                                title='Please select a model (.pth)')
        root.destroy()

        # best_model = torch.load(model_path) # This line doesn't work anymore because the model is now saved in a dict
        best_model_dict = torch.load(model_path)
        best_model = best_model_dict['model']
        CLASSES = best_model_dict['classes']
        print('model loaded!')

        test_dataset = USDataset(dfTest, classes=CLASSES,
                                 augmentation=pp.get_US_validation_augmentation(),
                                 preprocessing=pp.get_preprocessing(preprocessing_fn))

        # We have pre-processing here (to make sure the prediction works properly), but no need augmentation
        # because we're not training anymore
        test_dataloader = DataLoader(test_dataset)
        test_epoch = smp.utils.train.ValidEpoch(
            model=best_model,
            loss=best_model_dict['loss'],
            metrics=metrics,
            device=DEVICE,
        )

        if args.just_show is not None:
            num_to_display = args.just_show
        else:
            logs = test_epoch.run(test_dataloader)
            num_to_display = 2  # 2 cases but each class has C classes

        # # dataset for visualization without augmentation
        test_dataset_vis = USDataset(dfTest, classes=CLASSES,
                                     augmentation=pp.get_US_validation_augmentation())  # no preprocessing; this is the native image

        idxes = np.random.choice(len(test_dataset_vis), num_to_display, replace=False)
        for i in range(num_to_display):
            n = idxes[i]
            print(test_dataset_vis.images_fps[n])
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
                    fn=test_dataset_vis.images_fps[n],
                    cur_class=CLASSES[j],
                    image=image_vis,
                    gt=gt_mask,
                    pr=pr_mask,
                    iou=metrics[0].forward(torch.from_numpy(gt_mask), torch.from_numpy(pr_mask))
                )


if __name__ == '__main__':
    main()