use by:
```
python train_mine.py --data_path path/to/data --snapshot_path ./log --checkpoint ./sam_vit_b_01ec64.pth --lr 0.001 --rand_crop_size 128 --batch_size 1
```

problem:
train loss (using DiceCEloss) constantly at 0.5, val loss (using DiceLoss) constantly at 1.0, potentially means no intersection between label and prediction, and can't be properly trained.