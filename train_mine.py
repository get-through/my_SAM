import torch
import numpy as np
import torch.backends.cudnn as cudnn
import os
from torch.utils.tensorboard import SummaryWriter
import time
from pathlib import Path
import argparse
import logging
import shutil
from functools import partial
from segment_anything.utils.transforms import ResizeLongestSide
from torchvision.transforms import Resize
from torch.utils.tensorboard import SummaryWriter
from monai.metrics import DiceMetric

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from torch.optim import AdamW
import torch.nn as nn
import imageio

from monai.utils import first, set_determinism
from monai.losses import DiceCELoss, DiceLoss, FocalLoss
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
import matplotlib.pyplot as plt

organ_range = [
    (-714,502),
    (-90,112),
    (-75,91),
    (-103,99),
    (-80,122),
    (-56,96),
    (-156,338),
    (-714,176),
    (-130,315),
    (-86,122),
    (-91,115),
    (-75,91),
    (-72,85),
    (-646,166),
    (-116,502),
    (-57,154)
]

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
    parser.add_argument('--organ_id', default=1, type=int)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay (default: 0.001)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--num_points', default=4, type=int)

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--save_pic_dir', default='.', type=str,
                        help='picture saving path')

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
    filepath_last = os.path.join(checkpoint, f"last_{args.save_pic_dir}.pth")
    filepath_best = os.path.join(checkpoint, f"best_{args.save_pic_dir}.pth")
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

    os.makedirs(args.log_dir, exist_ok=True)

    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))
    log_writer = SummaryWriter(log_dir=args.log_dir)

    sam, msg = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    logger.info("checkpoint load msg:{}".format(msg))
    state_dict = torch.load("./log/best_pic7.pth")["dict"]
    msg = sam.load_state_dict(state_dict, strict=False)
    print(msg)
    sam.to(device)

    # fix the seed for reproducibility
    seed = args.seed
    organ_id = args.organ_id

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
                a_min=organ_range[organ_id][0],
                a_max=organ_range[organ_id][1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys=["image","label"], axcodes="RAS"),
            Spacingd(
                keys=["image","label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear","nearest"),
            ),
            CropForegroundd(keys=["image","label"], source_key="image"),
            # some label has one dim even <96
            AdaptiveRandCropByPosNegLabeld(
                keys=['image', 'label'],
                label_key='label',
                desired_spatial_size=(224, 224, 224),
                pos=2,
                neg=1,
                num_samples=1,
            ),
            # memory not enough, first save as 96, when training/val, resize to 224, possible resolution loss
            Resized(keys=["image","label"], spatial_size=96, mode=("bilinear","nearest"), size_mode="longest"),
            SpatialPadd(keys=["image","label"], spatial_size=(96,96,96), method="end"),
            # not working
            # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            EnsureTyped(keys=["image","label"], device=device, track_meta=False),
        ]
    )
    val_transforms = Compose(
        [   
            LoadImaged(keys=["image","label"]),
            EnsureChannelFirstd(keys=["image","label"]),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=organ_range[organ_id][0],
                a_max=organ_range[organ_id][1],
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Orientationd(keys=["image","label"], axcodes="RAS"),
            Spacingd(
                keys=["image","label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear","nearest"),
            ),
            CropForegroundd(keys=["image","label"], source_key="image"),
            Resized(keys=["image","label"], spatial_size=96, mode=("bilinear","nearest"), size_mode="longest"),
            SpatialPadd(keys=["image","label"], spatial_size=(96,96,96), method="end"),
            EnsureTyped(keys=["image","label"], device=device, track_meta=False),
        ]
    )

    train_images = sorted(glob.glob(os.path.join(args.data_path, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(args.data_path, "labelsTr", "*.nii.gz")))
    train_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
    val_images = sorted(glob.glob(os.path.join(args.data_path, "imagesVa", "*.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(args.data_path, "labelsVa", "*.nii.gz")))
    val_files = [{"image": image_name, "label": label_name} for image_name, label_name in zip(val_images, val_labels)]

    set_determinism(seed=0)
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=4)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=4)

    # print(first(train_ds))
    # print(first(train_ds)["image"].shape, first(train_ds)["image"].dtype)

    eff_batch_size = args.batch_size * args.accum_iter

    # only train on samples with the organ
    filtered_train_ds = [sample for sample in train_ds if torch.where(sample[0]["label"]==organ_id)[0].shape[0]>0]
    filtered_val_ds = [sample for sample in val_ds if torch.where(sample["label"]==organ_id)[0].shape[0]>0]

    print("train_ds from",len(train_ds),"to",len(filtered_train_ds))
    print("val_ds from",len(val_ds),"to",len(filtered_val_ds))

    data_loader_train = ThreadDataLoader(filtered_train_ds, num_workers=0, batch_size=eff_batch_size, shuffle=True, drop_last=True)
    data_loader_val = ThreadDataLoader(filtered_val_ds, num_workers=0, batch_size=eff_batch_size, shuffle=True, drop_last=True)

    # data_loader_train = ThreadDataLoader(train_ds, num_workers=0, batch_size=eff_batch_size, shuffle=True, drop_last=True)
    # data_loader_val = ThreadDataLoader(val_ds, num_workers=0, batch_size=eff_batch_size, shuffle=True, drop_last=True)

    # define the model

    opt = AdamW([i for i in sam.parameters() if i.requires_grad==True], lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.01, total_iters=500)
    
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="mean")
    loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.8, lambda_ce=0.2)
    focal_loss = FocalLoss(include_background=True,alpha=0.25,reduction="mean",use_softmax=True,weight=0.25,to_onehot_y=True)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    best_loss = np.inf
    flag = 1

    # times = 15
    weight_organ = 0.6
    weight_all = 0.4
    weight_bce = 0.5
    low = 0
    for epoch_num in range(args.max_epoch):
        loss_summary = []
        sam.train()
        for idx, data in enumerate(data_loader_train):
            resizer_image = Resize(spatial_size=224, mode=("bilinear"), size_mode="longest")
            resizer_label = Resize(spatial_size=224, mode=("nearest"), size_mode="longest")

            samples = [resizer_image(img) for img in data["image"]]
            samples = torch.stack(samples)
            img = samples.repeat(1,3,1,1,1).to(device)
            # print("data shape:",img.shape)
            segs = [resizer_label(img) for img in data["label"]]
            seg = torch.stack(segs)
            # print('seg:', seg[seg!=0])

            input_point = []

            label_point_pos = torch.nonzero(seg[0,0,:,:,:]==organ_id)
            print("label_point_length:",len(label_point_pos))
            # assert len(label_point_pos) >= args.num_points*5
            torch.manual_seed(42)
            if len(label_point_pos) > args.num_points:
                ind = torch.randperm(len(label_point_pos))[:args.num_points]
                input_point_pos = torch.stack([label_point_pos[ind[i]] for i in range(args.num_points)]).unsqueeze(0)

                label_point_neg = torch.where(seg[0,0,:,:,:]!=organ_id)
                ind = torch.randperm(label_point_neg[0].shape[0])[:args.num_points]
                input_point_neg = torch.stack([torch.tensor([label_point_neg[0][ind[i]],label_point_neg[1][ind[i]],label_point_neg[2][ind[i]]]) for i in range(args.num_points)]).unsqueeze(0)

                labels_torch_pos = torch.ones(1,args.num_points)
                labels_torch_neg = torch.zeros(1,args.num_points)

                print(labels_torch_pos.device,labels_torch_neg.device,input_point_pos.device,input_point_neg.device)
                labels_torch = torch.cat((labels_torch_pos,labels_torch_neg),dim=1).to(device)
                input_point = torch.cat((input_point_pos,input_point_neg.to(device)),dim=1)
            else:
                logger.info("has no organ label")
                label_point = torch.where(seg[0,0,:,:,:]==0)
                ind = torch.randperm(label_point_neg[0].shape[0])[:2*args.num_points]
                input_point = torch.stack([torch.tensor([label_point[0][ind[i]],label_point[1][ind[i]],label_point[2][ind[i]]]) for i in range(args.num_points)]).unsqueeze(0)

                labels_torch = torch.zeros(1,2*args.num_points)

            transform = ResizeLongestSide(sam.image_encoder.img_size)
            coords_torch = transform.apply_coords(input_point, samples.shape[-3:])
            # coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
            # labels_torch = torch.as_tensor(input_label, dtype=torch.int, device=device)
            # print("coords, labels:",coords_torch,labels_torch)
            # coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            points = (coords_torch, labels_torch)
            print("train coords, labels:",coords_torch.shape, labels_torch.shape)

            # input_batch = dict()
            # input_batch["image"] = samples[0]
            # input_batch["original_size"] = samples.shape[2:]
            # input_batch["point_coords"] = coords_torch
            # input_batch["point_labels"] = labels_torch
            # input_batch["boxes"] = None
            # input_batch["mask_inputs"] = None
            # batched_input = list()
            # batched_input.append(input_batch)
            # out = sam(batched_input,False)
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=points,
                    boxes=None,
                    masks=None,
            )
            masks, organ, iou_predictions = sam.mask_decoder(
                samples,
                image_embeddings=sam.image_encoder(img),
                image_pe=sam.prompt_encoder.get_dense_pe(),
                points = coords_torch,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            # masks = sam.postprocess_masks(low_res_masks, img.shape[-3:], img.shape[-3:])
            # organ = sam.postprocess_masks(low_res_organ, img.shape[-3:], img.shape[-3:])

            # seg[seg>0] = 1
            # masks[masks<0] = 0
            # masks[masks>0] = 1
            # masks = masks.clip(0,1)
            assert organ.shape == (1,2,224,224,224)
            print("mask:",masks.shape)

            seg = seg.to(device)

            assert torch.min(seg) >= 0

            seg[seg!=organ_id] = 0
            seg[seg>0] = 1
            
            # seg[seg>times] = 0
            diceloss = dice_loss(masks.clip(0,1), seg.clip(0,1))
            bce_loss = torch.nn.BCELoss()
            bceloss = bce_loss(masks.clip(0,1),seg.clip(0,1))

            # diceloss_organ = dice_loss(organ,seg)
            # if(low):
            #     organ = torch.argmax(organ,dim=1)
            #     organ = organ.unsqueeze(0)
            diceceloss_organ = loss_cal(organ.float(),seg)
            focalloss = focal_loss(organ,seg.clip(0,1))
            loss = weight_all * (diceloss + weight_bce * bceloss) + weight_organ * diceceloss_organ 

            # print("loss:",loss)
            loss_summary.append(loss.detach().cpu().numpy())
            opt.zero_grad()
            # print(loss.requires_grad, loss.dtype)
            # print(loss.grad_fn)
            loss.backward()
            logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num+1, args.max_epoch, idx+1, len(data_loader_train)) + ", lr:{}".format(opt.param_groups[0]['lr']) + ", loss:" + str(
                    loss_summary[-1].flatten()[0]) + ", dice_loss:" + str(diceloss.detach().cpu().numpy()) + ", bce_loss:" + str(bceloss.detach().cpu().numpy()) + ", organ_loss:" + str(diceceloss_organ.detach().cpu().numpy()) + ", focal_loss:" + str(focalloss.detach().cpu().numpy()))
            torch.nn.utils.clip_grad_norm_(sam.parameters(), 1.0)
            opt.step()
        scheduler.step()

        logger.info("- Train metrics: " + str(np.mean(loss_summary)))
        log_writer.add_scalar('train_loss', np.mean(loss_summary), epoch_num)
        log_writer.add_scalar('lr', opt.param_groups[0]['lr'], epoch_num)

        sam.eval()
        with torch.no_grad():
            loss_summary = []
            for idx, data in enumerate(data_loader_val):
                samples = [resizer_image(img) for img in data["image"]]
                samples = torch.stack(samples)
                img = samples.repeat(1,3,1,1,1).to(device)
                segs = [resizer_label(img) for img in data["label"]]
                seg = torch.stack(segs)
                # seg[seg>times] = 0
                # print('seg: ', seg.sum())
                input_point = []
                label_point = torch.nonzero(seg[0,0,:,:,:]==organ_id)
                print("label_point_length:",len(label_point))
                assert len(label_point) >= args.num_points*5
                torch.manual_seed(42)
                ind  = torch.randperm(len(label_point))[:args.num_points]
                input_point_pos = torch.stack([label_point[ind[i]] for i in range(args.num_points)]).unsqueeze(0)
                # print(input_point)

                label_point_neg = torch.where(seg[0,0,:,:,:]!=organ_id)
                ind = torch.randperm(label_point_neg[0].shape[0])[:args.num_points]
                input_point_neg = torch.stack([torch.tensor([label_point_neg[0][ind[i]],label_point_neg[1][ind[i]],label_point_neg[2][ind[i]]]) for i in range(args.num_points)]).unsqueeze(0)

                labels_torch_pos = torch.ones(1,args.num_points)
                labels_torch_neg = torch.zeros(1,args.num_points)
                labels_torch = torch.cat((labels_torch_pos,labels_torch_neg),dim=1).to(device)
                input_point = torch.cat((input_point_pos,input_point_neg.to(device)),dim=1)

                transform = ResizeLongestSide(sam.image_encoder.img_size)
                coords_torch = transform.apply_coords(input_point, img.shape[-3:])
                # coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=device)
                print("val coords, labels:",coords_torch.shape,labels_torch.shape)
                points = (coords_torch, labels_torch)
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=points,
                    boxes=None,
                    masks=None,
                )
                low_res_masks, low_res_organ, iou_predictions = sam.mask_decoder(
                    samples,
                    image_embeddings=sam.image_encoder(img),
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    points = coords_torch,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                masks = sam.postprocess_masks(low_res_masks, img.shape[-3:], img.shape[-3:])
                organ = sam.postprocess_masks(low_res_organ, img.shape[-3:], img.shape[-3:])

                # masks[masks>0] = 1
                # masks[masks<0] = 0
                
                seg = seg.to(device)
                seg[seg!=organ_id] = 0
                seg[seg>0]=1
                # masks = masks.to(device)
                organ = torch.argmax(organ,dim=1)
                organ = organ.unsqueeze(0)
                loss = dice_loss(organ, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num+1, args.max_epoch, idx+1, len(data_loader_val)) + ": loss:" + str(
                        loss_summary[-1].flatten()[0]))
        logger.info("- Val metrics: " + str(np.mean(loss_summary)))
        log_writer.add_scalar('val_loss', np.mean(loss_summary), epoch_num)

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
        log_writer.add_scalar('val_best', best_loss, epoch_num)

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
    def __init__(self, keys, label_key, desired_spatial_size, pos, neg, num_samples):
        self.keys = keys
        self.label_key = label_key
        self.desired_spatial_size = desired_spatial_size
        self.pos = pos
        self.neg = neg
        self.num_samples = num_samples

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
            neg=self.neg,
            num_samples = self.num_samples,
        )
        
        return rcrop(data)
    
# Define a function to save each slice as an image
def save_slice(slice_img, slice_label, index, save_dir):
    # fig = plt.figure(figsize=(10, 10))

    plt.imshow(slice_img, cmap='gray')
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = slice_label.shape[-2:]
    mask_image = slice_label.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax = plt.gca()
    ax.imshow(mask_image)
    plt.axis('off')

    plt.savefig(f"{save_dir}/slice_{index:03d}.png")

def save_video(image, label, frames_dir, epoch_num):
    # Save each slice to the frames directory
    # print("volume shape:",volume.shape)
    for (i, slice_img), (j, slice_label) in zip(enumerate(image),enumerate(label)):
        save_slice(slice_img, slice_label, i, frames_dir)

    # List out the saved frames in order
    frame_files = [f"{frames_dir}/slice_{i:03d}.png" for i in range(image.shape[0])]

    # Create a video using imageio
    video_path = f'{frames_dir}/slice_visualization_{epoch_num}.mp4'
    writer = imageio.get_writer(video_path, fps=10)  # fps controls the speed of the video

    for frame_file in frame_files:
        writer.append_data(imageio.imread(frame_file))
    writer.close()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.snapshot_path:
        Path(args.snapshot_path).mkdir(parents=True, exist_ok=True)
    main(args)
