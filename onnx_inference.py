#-*-coding:utf-8-*-
# date:2021-10-5
# Author: Eric.Lee
# function: onnx Inference
import os, sys
sys.path.append(os.getcwd())
import onnxruntime
import onnx
import cv2
import torch
import numpy as np
from hand_data_iter.datasets import draw_bd_handpose
class ONNXModel():
    def __init__(self, onnx_path,gpu_cfg = False):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        if gpu_cfg:
            self.onnx_session.set_providers(['CUDAExecutionProvider'], [ {'device_id': 0}])
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed

    def forward(self, image_numpy):
        '''
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        input_feed = self.get_input_feed(self.input_name, image_numpy)
        output = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return output
if __name__ == "__main__":
    img_size = 256
    model = ONNXModel("resnet_50_size-256.onnx")
    path_ = "./image/"
    for f_ in os.listdir(path_):

        img0 = cv2.imread(path_ + f_)
        img_width = img0.shape[1]
        img_height = img0.shape[0]
        img = cv2.resize(img0, (img_size,img_size), interpolation = cv2.INTER_CUBIC)

        img_ndarray = img.transpose((2, 0, 1))
        img_ndarray = img_ndarray / 255.
        img_ndarray = np.expand_dims(img_ndarray, 0)

        output = model.forward(img_ndarray.astype('float32'))[0][0]
        output = np.array(output)
        print(output.shape[0])
        pts_hand = {} #构建关键点连线可视化结构
        for i in range(int(output.shape[0]/2)):
            x = (output[i*2+0]*float(img_width))
            y = (output[i*2+1]*float(img_height))

            pts_hand[str(i)] = {}
            pts_hand[str(i)] = {
                "x":x,
                "y":y,
                }

        draw_bd_handpose(img0,pts_hand,0,0) # 绘制关键点连线

        #------------- 绘制关键点
        for i in range(int(output.shape[0]/2)):
            x = (output[i*2+0]*float(img_width))
            y = (output[i*2+1]*float(img_height))

            cv2.circle(img0, (int(x),int(y)), 3, (255,50,60),-1)
            cv2.circle(img0, (int(x),int(y)), 1, (255,150,180),-1)


        cv2.namedWindow('image',0)
        cv2.imshow('image',img0)
        if cv2.waitKey(600) == 27 :
            break

        cv2.waitKey(0)
