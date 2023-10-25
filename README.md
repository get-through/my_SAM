use by:
```
python train_mine.py --data_path path/to/data --snapshot_path ./log --checkpoint ./sam_vit_b_01ec64.pth --lr 0.001 --rand_crop_size 128 --batch_size 1
```

problem probably be related to:
1. through the process of the model, some new variables are produced
2. the produced mask is in dtype Bool
3. other reason

currently set dataset to accept 1 data, just to make it convenient for debugging.