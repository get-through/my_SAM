use by:
```
python train_mine.py --data_path ../../../data/xinglujing/data/nnFormer_raw/Dataset500_amos/ --snapshot_path ./organs_log --checkpoint ./sam_vit_b_01ec64.pth --lr 1e-4 --rand_crop_size 128 --batch_size 1 --save_pic_dir organ3 --num_points 4 --organ_id 3
```
checkpoint is where to load vit_b's path from
organ_id is to set which organ to train on
save_pic_dir is to specify the name of path, i.e. best_name.pth, last_name.pth
num_points is to specify how many points to send into prompt encoder, currently will result in N pos(organ) points and N neg(non-organ) points to send into the prompt.
snapshot_path is where to save the train_log.txt and the paths

"./log/best_pic7.pth" is a pretrained weight on all organs, letting the model to initially attend to the organ area