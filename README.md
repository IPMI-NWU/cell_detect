#  Predicting the Cell-level PD-L1 Expression Status from H&E-stained Histopa thological Slide

# Envs
- Linux
- Python>=3.7
- CPU or NVIDIA GPU + CUDA CuDNN
- openslide

# datasets:  Private dataset


# Train

```
python train.py  --output_dir=outmodel.pth --eos_coef=0.8 --dataset=path/to/dataset/4class/ --num_classes=4 --num_workers=4 --start_eval=1 --epochs=200 --batch_size=4


