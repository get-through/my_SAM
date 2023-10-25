import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from torch.utils.tensorboard import SummaryWriter
import time
from pathlib import Path
import random
import json
import datetime
import argparse
import torch.nn.functional as F
import logging
import shutil
from functools import partial
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms import Resize

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from torch.optim import AdamW

from monai.utils import first, set_determinism
from monai.losses import DiceCELoss, DiceLoss
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    SpatialPadd,
    Resized,
    Transform,
    Resize,
)
from monai.data import CacheDataset, ThreadDataLoader
import glob
from collections import OrderedDict
import matplotlib.pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser('SAM model finetuning', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--rand_crop_size', nargs='+', type=int, default=224,
                        help='patch size for later prompt use')
    parser.add_argument('--max_epoch', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default="vit_b", type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay (default: 0.001)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--model_type', default='vit_b', help='type of the model, vit_b/l/h')
    parser.add_argument('--checkpoint', default='.', help='checkpoint of sam')
    parser.add_argument('--snapshot_path', default='./', help='save directory for snapshots')

    return parser

def save_checkpoint(state, is_best, checkpoint):
    filepath_last = os.path.join(checkpoint, "last.pth.tar")
    filepath_best = os.path.join(checkpoint, "best.pth.tar")
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Masking directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint DIrectory exists!")
    torch.save(state, filepath_last)
    if is_best:
        if os.path.isfile(filepath_best):
            os.remove(filepath_best)
        shutil.copyfile(filepath_last, filepath_best)

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    train_transforms = Compose(
        [   
            LoadImaged(keys=["image","label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys=["image","label"], axcodes="RAS"),
            Spacingd(
                keys=["image","label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear"),
            ),
            CropForegroundd(keys=["image","label"], source_key="image"),
            # AdaptiveRandCropByPosNegLabeld(
            #     keys=['image', 'label'],
            #     label_key='label',
            #     desired_spatial_size=(96, 96, 96),
            #     pos=1,
            #     neg=0
            # ),
            Resized(keys=["image","label"], spatial_size=96, mode=("bilinear"), size_mode="longest"),
            SpatialPadd(keys=["image","label"], spatial_size=(96,96,96), method="end"),
            EnsureTyped(keys=["image","label"], device=device, track_meta=False),
        ]
    )
    val_transforms = Compose(
        [   
            LoadImaged(keys=["image","label"]),
            EnsureChannelFirstd(keys=["image","label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-57,
                a_max=164,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys=["image","label"], axcodes="RAS"),
            Spacingd(
                keys=["image","label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear"),
            ),
            CropForegroundd(keys=["image","label"], source_key="image"),
            # Resized(keys=["image"], spatial_size=(224,224,64), mode=("bilinear")),
            Resized(keys=["image","label"], spatial_size=96, mode=("bilinear"), size_mode="longest"),
            SpatialPadd(keys=["image","label"], spatial_size=(96,96,96), method="end"),
            # Resized(keys=["image"], spatial_size=(224,128,224), mode=("bilinear")),
            EnsureTyped(keys=["image","label"], device=device, track_meta=False),
        ]
    )

    train_images = sorted(glob.glob(os.path.join(args.data_path, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.data_path, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    index = [2,12,32,42,52,62,72,82,92,102,112,122,132,142,152,162,172,182,192,202,212,222,232,242,252,262,272,282,292,302,312,322,332,342,352,362,372,382,392,402.412,422,432,442,452,462,472,482,492]
    j=0
    k=0
    train_files = dict()
    val_files = dict()
    # for i in range(len(data_dicts)):
    #     if i not in index:
    #         train_files[j] = data_dicts[i]
    #         j += 1
    #     else:
    #         val_files[k] = data_dicts[i]
    #         k += 1
    
    train_files[0] = data_dicts[0]
    val_files[0] = data_dicts[0]

    set_determinism(seed=0)
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)

    # print(first(train_ds))
    # print(first(train_ds)["image"].shape, first(train_ds)["image"].dtype)

    os.makedirs(args.log_dir, exist_ok=True)

    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))

    eff_batch_size = args.batch_size * args.accum_iter

    for sample in train_ds:
        print(sample)
        break

    filtered_train_ds = [sample for sample in train_ds if sample["label"].sum()>0]
    filtered_val_ds = [sample for sample in val_ds if sample["label"].sum()>0]

    print("train_ds from",len(train_ds),"to",len(filtered_train_ds))
    print("val_ds from",len(val_ds),"to",len(filtered_val_ds))

    data_loader_train = ThreadDataLoader(filtered_train_ds, num_workers=0, batch_size=eff_batch_size, shuffle=True, drop_last=True)
    data_loader_val = ThreadDataLoader(filtered_val_ds, num_workers=0, batch_size=eff_batch_size, shuffle=True, drop_last=True)

    check_data = first(data_loader_train)
    print(check_data.keys())

    # define the model
    sam, msg = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    logger.info("checkpoint load msg:{}".format(msg))
    sam.to(device)

    opt = AdamW([i for i in sam.parameters() if i.requires_grad==True], lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.01, total_iters=500)
    
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    best_loss = np.inf

    for epoch_num in range(args.max_epoch):
        loss_summary = []
        sam.train()
        for idx, data in enumerate(data_loader_train):
            resizer = Resize(spatial_size=224, mode=("bilinear"), size_mode="longest")
            samples = [resizer(img) for img in data["image"]]
            samples = torch.stack(samples)
            samples = samples.repeat(1,3,1,1,1).to(device)
            segs = [resizer(img) for img in data["label"]]
            seg = torch.stack(segs)
            # print('seg:', seg[seg!=0])

            input_point = []
            label_point = torch.nonzero(seg[0,0,:,:,:]==1)
            # print(label_point)
            input_point = label_point[len(label_point)//2].unsqueeze(0)
            # print(input_point)
            input_label = np.array([1])
            transform = ResizeLongestSide(sam.image_encoder.img_size)
            coords_torch = transform.apply_coords(input_point, samples.shape[-3:])
            # coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
            labels_torch = torch.as_tensor(input_label, dtype=torch.int, device=device)
            print("coords, labels:",coords_torch,labels_torch)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            points = (coords_torch, labels_torch)

            input_batch = dict()
            input_batch["image"] = samples[0]
            input_batch["original_size"] = samples.shape[2:]
            input_batch["point_coords"] = coords_torch
            input_batch["point_labels"] = labels_torch
            input_batch["boxes"] = None
            input_batch["mask_inputs"] = None
            batched_input = list()
            batched_input.append(input_batch)
            out = sam(batched_input,False)

            seg = seg.to(device)
            masks = out[0]["masks"]
            print("mask and seg shape:", masks.shape, seg.shape)
            loss = loss_cal(masks.to(dtype=torch.float), seg)
            print("loss:",loss)
            loss_summary.append(loss.detach().cpu().numpy())
            opt.zero_grad()
            # loss.requires_grad_()
            loss = loss.to(dtype=torch.float)
            print(loss.requires_grad, loss.dtype)
            print(hasattr(loss, 'grad_fn'))
            loss.backward()
            logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(data_loader_train)) + ", lr:{}".format(opt.param_groups[0]['lr']) + ", loss:" + str(
                    loss_summary[-1].flatten()[0]))
            torch.nn.utils.clip_grad_norm_(sam.parameters(), 1.0)
            opt.step()
        scheduler.step()

        logger.info("- Train metrics: " + str(np.mean(loss_summary)))

        sam.eval()
        with torch.no_grad():
            loss_summary = []
            for idx, data in enumerate(data_loader_val):
                samples = [resizer(img) for img in data["image"]]
                samples = torch.stack(samples)
                img = samples.repeat(1,3,1,1,1).to(device)
                segs = [resizer(img) for img in data["label"]]
                seg = torch.stack(segs)
                # print('seg: ', seg.sum())
                input_point = []
                label_point = torch.nonzero(seg[0,0,:,:,:]==1)
                # print(label_point)
                input_point = label_point[len(label_point)//2].unsqueeze(0)
                # print(input_point)
                input_label = np.array([1])
                transform = ResizeLongestSide(sam.image_encoder.img_size)
                coords_torch = transform.apply_coords(input_point, img.shape[-3:])
                # coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
                labels_torch = torch.as_tensor(input_label, dtype=torch.int, device=device)
                print("coords, labels:",coords_torch,labels_torch)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                points = (coords_torch, labels_torch)
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=points,
                    boxes=None,
                    masks=None,
                )
                low_res_masks, iou_predictions = sam.mask_decoder(
                    image_embeddings=sam.image_encoder(img),
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                masks = sam.postprocess_masks(low_res_masks, img.shape[-3:], img.shape[-3:])
                masks = masks > 0
                
                seg = seg.to(device)
                masks = masks.to(device, dtype=torch.float)
                loss = dice_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(data_loader_val)) + ": loss:" + str(
                        loss_summary[-1].flatten()[0]))
        logger.info("- Val metrics: " + str(np.mean(loss_summary)))

        is_best = False
        if np.mean(loss_summary) < best_loss:
            best_loss = np.mean(loss_summary)
            is_best = True
        save_checkpoint({"epoch": epoch_num + 1,
                        "best_val_loss": best_loss,
                         "dict": sam.state_dict(),
                         "opt": opt.state_dict(),
                         },
                        is_best=is_best,
                        checkpoint=args.snapshot_path)
        logger.info("- Val metrics best: " + str(best_loss))

def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "{}_{}.log".format(logger_name, get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg

def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime

class AdaptiveRandCropByPosNegLabeld(Transform):
    def __init__(self, keys, label_key, desired_spatial_size, pos, neg):
        self.keys = keys
        self.label_key = label_key
        self.desired_spatial_size = desired_spatial_size
        self.pos = pos
        self.neg = neg

    def __call__(self, data):
        # Calculate the adaptive spatial size
        image_size = data[self.keys[0]].shape[1:]  # Assuming CHWD format
        spatial_size = tuple(min(d, s) for d, s in zip(self.desired_spatial_size, image_size))
        
        # Apply the RandCropByPosNegLabeld transform
        rcrop = RandCropByPosNegLabeld(
            keys=self.keys,
            label_key=self.label_key,
            spatial_size=spatial_size,
            pos=self.pos,
            neg=self.neg
        )
        
        return rcrop(data)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.snapshot_path:
        Path(args.snapshot_path).mkdir(parents=True, exist_ok=True)
    main(args)
