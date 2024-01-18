# [原始的README](./old_README.md)

https://github.com/baegwangbin/surface_normal_uncertainty

# 训练和测试

## 准备数据集

准备训练集：  
1、参考 `VCCPlaneRecon\datasets\generator\capture.py` 采集数据。   
2、参考`data\dataloader_sz.py` 。

准备测试集，参考`collect_imgs_dataset.py` 。

## 训练

`train.py --pretrained scannet --architecture BN --n_epochs 10 --lr 3e-4`

> 训练只在建筑物的遮罩上面进行，忽略天空和地面。

## 测试

`test.py --pretrained [path to model weights] --architecture BN --imgs_dir [path to test data]`

## 导出onnx模型

`export_onnx.py --pretrained [path to model weights] --architecture BN --imgs_dir [path to test data] --export_onnx`
> 测试导出后onnx推理功能：  
> `export_onnx.py --pretrained [path to model weights] --architecture BN --imgs_dir [path to test data]`


# 推理注意事项

## 图片预处理

如果在训练的时候启用 `use_clahe` 功能，推理时也需要用 `opencv.clache` 对图像进行预处理。

## 推理时间

使用`tf_efficientnet_lite0`后，模型需要400ms CPU时间。

> 如果使用`tf_mobilenetv3_small_minimal_100`，猜测模型甚至可以达到200ms CPU时间。

# TODO

- [x] 保证采样器只在有效遮罩内部采样
- [ ] 解码器速度太慢
