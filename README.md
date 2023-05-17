# [原始的README](./original_ReadMe.md)

# 训练和测试

## 准备数据集

准备训练集，参考`data\dataloader_sz.py` 。

准备测试集，参考`collect_imgs_dataset.py` 。

## 训练

`train.py --pretrained scannet --architecture BN --n_epochs 10 --lr 3e-4`

## 测试

`test.py --pretrained [path to model weights] --architecture BN --imgs_dir [path to test data]`

## 导出onnx模型

`export_onnx.py --pretrained [path to model weights] --architecture BN --imgs_dir [path to test data] --export_onnx`
> 测试导出后onnx推理功能：  
> `export_onnx.py --pretrained [path to model weights] --architecture BN --imgs_dir [path to test data]`