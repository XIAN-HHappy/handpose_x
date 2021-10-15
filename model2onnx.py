#-*-coding:utf-8-*-
# date:2021-10-5
# Author: Eric.Lee
# function: pytorch model 2 onnx

import os
import argparse
import torch
import torch.nn as nn
import numpy as np

from models.resnet import resnet18,resnet34,resnet50,resnet101
from models.squeezenet import squeezenet1_1,squeezenet1_0
from models.shufflenetv2 import ShuffleNetV2
from models.shufflenet import ShuffleNet
from models.mobilenetv2 import MobileNetV2
from torchvision.models import shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0
from models.rexnetv1 import ReXNetV1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=' Project Hand Pose Inference')
    parser.add_argument('--model_path', type=str, default = './weights1/resnet_50-size-256-wingloss102-0.119.pth',
        help = 'model_path') # 模型路径
    parser.add_argument('--model', type=str, default = 'shufflenet_v2_x1_5',
        help = '''model : resnet_34,resnet_50,resnet_101,squeezenet1_0,squeezenet1_1,shufflenetv2,shufflenet,mobilenetv2
            shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0''') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 42,
        help = 'num_classes') #  手部21关键点， (x,y)*2 = 42
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
    parser.add_argument('--test_path', type=str, default = './image/',
        help = 'test_path') # 测试图片路径
    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--vis', type=bool , default = True,
        help = 'vis') # 是否可视化图片

    print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    ops = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    print('----------------------------------')

    unparsed = vars(ops) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))

    #---------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS

    test_path =  ops.test_path # 测试图片文件夹路径

    #---------------------------------------------------------------- 构建模型
    print('use model : %s'%(ops.model))

    if ops.model == 'resnet_50':
        model_ = resnet50(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == 'resnet_18':
        model_ = resnet18(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == 'resnet_34':
        model_ = resnet34(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == 'resnet_101':
        model_ = resnet101(num_classes = ops.num_classes,img_size=ops.img_size[0])
    elif ops.model == "squeezenet1_0":
        model_ = squeezenet1_0(num_classes=ops.num_classes)
    elif ops.model == "squeezenet1_1":
        model_ = squeezenet1_1(num_classes=ops.num_classes)
    elif ops.model == "shufflenetv2":
        model_ = ShuffleNetV2(ratio=1., num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x1_5":
        model_ = shufflenet_v2_x1_5(pretrained=False,num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x1_0":
        model_ = shufflenet_v2_x1_0(pretrained=False,num_classes=ops.num_classes)
    elif ops.model == "shufflenet_v2_x2_0":
        model_ = shufflenet_v2_x2_0(pretrained=False,num_classes=ops.num_classes)
    elif ops.model == "shufflenet":
        model_ = ShuffleNet(num_blocks = [2,4,2], num_classes=ops.num_classes, groups=3)
    elif ops.model == "mobilenetv2":
        model_ = MobileNetV2(num_classes=ops.num_classes)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval() # 设置为前向推断模式

    # 加载测试模型
    if os.access(ops.model_path,os.F_OK):# checkpoint
        chkpt = torch.load(ops.model_path, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.model_path))

    input_size = ops.img_size[0]
    batch_size = 1  #批处理大小
    input_shape = (3, input_size,input_size)   #输入数据,改成自己的输入shape
    print("input_size : ",input_size)

    x = torch.randn(batch_size, *input_shape)   # 生成张量
    x = x.to(device)
    export_onnx_file = "{}_size-{}.onnx".format(ops.model,input_size)		# 目的ONNX文件名
    torch.onnx.export(model_,
                        x,
                        export_onnx_file,
                        opset_version=9,
                        do_constant_folding=True,	# 是否执行常量折叠优化
                        input_names=["input"],	# 输入名
                        output_names=["output"],	# 输出名
                        #dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                        #                "output":{0:"batch_size"}}
                        )
