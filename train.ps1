$env:PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb=512'
python train.py --pretrained checkpoints/scannet.pt --architecture BN --n_epochs 10 --workers 4 --lr 0.00357 --exp_name exp01