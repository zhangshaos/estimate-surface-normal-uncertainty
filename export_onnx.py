import argparse
import os
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.onnx
import onnx
import onnxruntime as ort

from data.dataloader_sz import SZLoader
from models.NNET import NNET
import mainly_utils.utils as utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def save_normal_result(norm_out: np.ndarray, result_dir: str, img_base_name: str):
    alpha_max = 90
    pred_norm = norm_out[:, :3, :, :]
    pred_kappa = norm_out[:, 3:, :, :]
    pred_norm = pred_norm.transpose((0, 2, 3, 1)) # (B, H, W, 3)
    pred_kappa = pred_kappa.transpose((0, 2, 3, 1))

    # 2. predicted normal
    # 模型使用左x，上y，后z左手坐标系，将其转换为右x，下y，前z相机坐标系
    pred_norm *= -1
    pred_norm /= np.sum(np.square(pred_norm), axis=3, keepdims=True)
    pred_norm_rgb = ((pred_norm + 1) * 0.5) * 255
    pred_norm_rgb = np.clip(pred_norm_rgb, a_min=0, a_max=255)
    pred_norm_rgb = pred_norm_rgb.astype(np.uint8)  # (B, H, W, 3)

    target_path = f'{result_dir}/{img_base_name}_pred_norm.png'
    plt.imsave(target_path, pred_norm_rgb[0, :, :, :])

    # 4. predicted uncertainty
    pred_alpha = utils.kappa_to_alpha(pred_kappa)
    pred_alpha = np.clip(pred_alpha, 0, alpha_max)
    pred_alpha_gray = (pred_alpha * (255 / 90)).astype(np.uint8)

    target_path = f'{result_dir}/{img_base_name}_pred_alpha.png'
    # plt.imsave(target_path, pred_alpha_gray[0, :, :, 0], cmap='gray')
    Image.fromarray(pred_alpha_gray[0, :, :, 0]).save(target_path)


#https://github.com/microsoft/onnxruntime/blob/main/tools/python/remove_initializer_from_input.py
def remove_initializer_from_input(onnx_model_path: str) -> bool:
    model = onnx.load(onnx_model_path)
    if model.ir_version < 4:
        print("Model with ir_version below 4 requires to include initilizer in graph input")
        return False

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(model, onnx_model_path)
    return True


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@', conflict_handler='resolve')
    parser.convert_arg_line_to_args = utils.convert_arg_line_to_args

    parser.add_argument('--architecture', required=True, type=str, help='{BN, GN}')
    parser.add_argument("--pretrained", required=True, type=str, help="{pretrained model path}")
    parser.add_argument('--sampling_ratio', type=float, default=0.4)
    parser.add_argument('--importance_ratio', type=float, default=0.7)
    parser.add_argument('--input_height', default=480, type=int)
    parser.add_argument('--input_width', default=640, type=int)
    parser.add_argument('--imgs_dir', required=True, type=str)
    parser.add_argument('--use_clahe', default=False, action="store_true")
    parser.add_argument('--export_onnx', default=False, action="store_true")

    # read arguments from txt file
    if sys.argv.__len__() == 2 and '.txt' in sys.argv[1]:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    # 导出为onnx模型
    export_onnx = args.export_onnx
    export_onnx = False
    if export_onnx:
        # load checkpoint
        checkpoint = args.pretrained
        print(f'loading checkpoint... {checkpoint}')
        model = NNET(args)
        model = utils.load_checkpoint(checkpoint, model)
        with torch.no_grad():
            model.eval()
            print('loading checkpoint... / done')
            print(f'export onnx model...')
            x = torch.randn(1, 3, 480, 640)
            model(x) #预热
            t0 = time.time_ns()
            out_list, _, _ = model(x)
            t1 = time.time_ns()
            print(f'pytorch inference cost time: {(t1 - t0) * 1e-6} ms.')
            norm_out = out_list[-1].numpy()
            onnx_model_name = 'nnet.onnx'
            torch.onnx.export(model,
                              x,
                              onnx_model_name,
                              verbose=False,
                              input_names=['input'],
                              output_names=[f'output{i}' for i in range(12)])
            time.sleep(1)
            print(f'export onnx model... / done')
            #Q：Why is the model graph not optimized even with graph_optimization_level set to ORT_ENABLE_ALL?
            #A：https://onnxruntime.ai/docs/performance/tune-performance/troubleshooting.html
            #remove_initializer_from_input(onnx_model_name)
            #time.sleep(1)
        #测试1
        onnx_nnet = onnx.load(onnx_model_name)
        onnx.checker.check_model(onnx_nnet)
        #测试2
        print(f'onnx runtime device: {ort.get_device()}')
        ort_session = ort.InferenceSession(onnx_model_name)
        ort_inputs = {'input': x.numpy()}
        ort_session.run(None, ort_inputs) #预热
        t0 = time.time_ns()
        ort_outs = ort_session.run(None, ort_inputs)
        t1 = time.time_ns()
        print(f'onnx inference cost time: {(t1-t0)*1e-6} ms.')
        ort_norm_out = ort_outs[3]
        diff = np.abs(norm_out - ort_norm_out).sum()
        print(f'difference is {diff}, mean is {diff / norm_out.size}.')
    else:
        onnx_model_name = 'nnet.onnx'
        if not os.path.exists(onnx_model_name):
            print(f'Not found `{onnx_model_name}`. Use --export_onnx to export `{onnx_model_name}` first.')
            exit(1)
        #测试集选择一张图片进行测试
        results_dir = f'{args.imgs_dir}/results'
        os.makedirs(results_dir, exist_ok=True)
        test_loader = SZLoader(args, 'test').data
        test_data = next(iter(test_loader))
        img = test_data['img']
        img_name = test_data['img_name'][0]
        print(f'test image path is: {img_name}')
        print(f'onnx runtime device: {ort.get_device()}')
        opt = ort.SessionOptions()
        #opt.enable_profiling = True
        opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        #opt.optimized_model_filepath = f'optmized_{onnx_model_name}'
        ort_session = ort.InferenceSession(onnx_model_name, opt)
        ort_input = {'input': img.numpy()}
        t0 = time.time_ns()
        ort_outs = ort_session.run(None, ort_input)
        t1 = time.time_ns()
        print(f'onnx inference cost time: {(t1 - t0) * 1e-6} ms.')
        norm_out = ort_outs[3]
        img_base_name = os.path.basename(img_name).replace('.png', '')
        save_normal_result(norm_out, results_dir, img_base_name)
        print(f'save test result in {results_dir}')

