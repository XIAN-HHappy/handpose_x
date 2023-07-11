# HandPose X  
手势 21 个关键点检测  ， handpose
## 重要更新
### 添加 onnx runtime C++ 项目示例支持CPU&GPU，对应项目地址：
* https://gitcode.net/EricLee/onnx_run  
* https://github.com/EricLee2021-72324/onnx_run
### 添加 onnx 模块，预训练模型中有转好的resnet50-onnx模型，注意：目前不支持rexnetv1
### "dpcas" 项目地址：https://codechina.csdn.net/EricLee/dpcas

## 项目 - 首发布地址  
### https://codechina.csdn.net/EricLee/handpose_x  
## 项目Wiki
### 注意：项目的相关资料信息或是更新状态会同步到项目Wiki，以便更好的对该项目进行维护和不断发展，谢谢大家对该项目的关注！

## 项目介绍   
注意：该项目不包括手部检测部分，手部检测项目地址：https://codechina.csdn.net/EricLee/yolo_v3   
该项目是对手的21个关键点进行检测，示例如下 ：    
* 图片示例：  
![image](https://github.com/EricLee2021-72324/handpose_x/raw/main/samples/test.png)    
* 视频示例：  
![video](https://github.com/EricLee2021-72324/handpose_x/raw/main/samples/sample.gif)    

## Demo小样    
* 示例1 - 按键操作     
  因为考虑到目前没有三维姿态不好识别按键按下三维动作，所以目前采用二维方式。    
  该示例的原理：通过简单的IOU跟踪，对二维目标如手的边界框或是特定手指的较长时间位置稳定性判断确定触发按键动作的时刻，用特定指尖的二维坐标确定触发位置。    
  （注意：目前示例并未添加到工程，后期整理后会进行发布，只是一个样例，同时希望同学们自己尝试写自己基于该项目的小应用。）     
![keyboard](https://github.com/EricLee2021-72324/handpose_x/raw/main/samples/keyboard.gif)  

* 示例2 - 手势交互：指定区域物体识别      
  该示例的出发点是希望通过手势指定用户想要识别的物体。那么就要选中物体的准确边界框才能达到理想识别效果。如果待识别目标边界框太大会引入背景干扰，太小又会时目标特征不完全。所以希望通过手势指定较准确的目标边界框。因为边界框涉及左上、右下两个二维坐标，所以通过两只手的特定指尖来确定。且触发逻辑与示例1相同。           
  该示例的原理：通过简单的IOU跟踪，对二维目标如手的边界框或是特定手指的较长时间位置稳定性判断确定触发按键动作的时刻，用特定指尖的二维坐标确定触发位置。         
  （注意：目前示例并未添加到工程，后期整理后会进行发布，只是一个样例，同时希望同学们自己尝试写自己基于该项目的小应用。）     
  该示例依赖于另外一个物体识别分类项目。  

![keyboard](https://github.com/EricLee2021-72324/handpose_x/raw/main/samples/recognize_obj0.gif)    

* 以下是对书上狗的图片进行分类识别的样例，同学们可以根据自己对应的物体识别分类需求替换对应的分类识别模型即可。    

![recoobj_book](https://github.com/EricLee2021-72324/handpose_x/raw/main/samples/recobj_book.gif)    
* [该Demo完整视频](https://www.bilibili.com/video/BV1nb4y1R7Zh/)       

该物体识别分类项目的地址为：  https://codechina.csdn.net/EricLee/classification     

* 示例3 - 静态手势     
  通过手关键点的二维角度约束关系定义静态手势。  
  示例中手势包括：fist five gun love one six three thumbup yeah    
  目前该示例由于静态手势数据集的限制，目前用手骨骼的二维角度约束定义静态手势，原理如下图,计算向量AC和DE的角度，它们之间的角度大于某一个角度阈值（经验值）定义为弯曲，小于摸一个阈值（经验值）为伸直。    
  注：这种静态手势识别的方法具有局限性，有条件还是通过模型训练的方法进行静态手势识别。   

![gs](https://github.com/EricLee2021-72324/handpose_x/raw/main/samples/gest.jpg)     

  视频示例如下图：   
![gesture](https://github.com/EricLee2021-72324/handpose_x/raw/main/samples/gesture.gif)     

* 示例4 - 静态手势交互（识别）      
  通过手关键点的二维角度约束关系定义静态手势。     
  该项目通过手势操作选择分类识别区域或是ocr识别区域，送入分类识别网络或是第三方web识别服务，亦或是检索数据库等应用。   

  原理：通过二维约束获得静态手势，该示例是通过 食指伸直（one） 和 握拳（fist）分别代表范围选择和清空选择区域。    
  建议最好还是通过分类模型做静态手势识别鲁棒和准确高，目前局限于静态手势训练集的问题用二维约束关系定义静态手势替代。    

![ocrreco](https://github.com/EricLee2021-72324/handpose_x/raw/main/samples/ocrreco.gif)       
* [该Demo完整视频](https://www.bilibili.com/video/BV1Bb4y1R7sd/)       

## 项目配置  
### 1、软件  
* 作者开发环境：  
* Python 3.7  
* PyTorch >= 1.5.1  
* opencv-python  
### 2、硬件  
* 普通USB彩色（RGB）网络摄像头    

## 数据集   
该数据集包括网络图片及数据集<<Large-scale Multiview 3D Hand Pose Dataset>>筛选动作重复度低的部分图片，进行制作(如有侵权请联系删除)，共49062个样本。         
<<Large-scale Multiview 3D Hand Pose Dataset>>数据集，其官网地址 http://www.rovit.ua.es/dataset/mhpdataset/       
感谢《Large-scale Multiview 3D Hand Pose Dataset》数据集贡献者：Francisco Gomez-Donoso, Sergio Orts-Escolano, and Miguel Cazorla. "Large-scale Multiview 3D Hand Pose Dataset". ArXiv e-prints 1707.03742, July 2017.    

* 标注文件示例：   
![label](https://github.com/EricLee2021-72324/handpose_x/raw/main/samples/label.png)   

* [该项目用到的制作数据集下载地址(百度网盘 Password: ara8 )](https://pan.baidu.com/s/1KY7lAFXBTfrFHlApxTY8NA)   

* 如果使用该数据集并发布相关项目或网络资源文章等，请讲述其数据集的出处 "https://codechina.csdn.net/EricLee/handpose_x"    
* 数据集读取脚本为：read_datasets.py,并需要相应更改脚本中的数据集路径。  

## 模型   
### 1、目前支持的模型 (backbone)

- [x] resnet18 & resnet34 & resnet50 & resnet101
- [x] squeezenet1_0 & squeezenet1_1
- [x] ShuffleNet & ShuffleNetV2
- [x] MobileNetV2
- [x] rexnetv1
- [x] shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0 (torchvision 版本)

### 2、预训练模型   

* [预训练模型下载地址(百度网盘 Password: 99f3 )](https://pan.baidu.com/s/1Ur6Ikp31XGEuA3hQjYzwIw)        


## 项目使用方法  
### 模型训练  
* 根目录下运行命令： python train.py       (注意脚本内相关参数配置 )   

### 模型推理  
* 根目录下运行命令： python inference.py        (注意脚本内相关参数配置 )   

### onnx使用  
* step1: 设定相关配置包括模型类型和模型参数路径，根目录下运行命令： python model2onnx.py        (注意脚本内相关参数配置 )
* step2: 设定onnx模型路径，根目录下运行命令： python onnx_inference.py   (注意脚本内相关参数配置 )
* 建议    
```

检测手bbox后，进行以下的预处理，crop手图片送入手关键点模型进行推理，   
可以参考 hand_data_iter/datasets.py,数据增强的样本预处理代码部分，   
关键代码如下：     
  img 为原图  ，np为numpy  
  x_min,y_min,x_max,y_max,score = bbox  
  w_ = max(abs(x_max-x_min),abs(y_max-y_min))  

  w_ = w_*1.1  

  x_mid = (x_max+x_min)/2  
  y_mid = (y_max+y_min)/2  

  x1,y1,x2,y2 = int(x_mid-w_/2),int(y_mid-w_/2),int(x_mid+w_/2),int(y_mid+w_/2)  

  x1 = np.clip(x1,0,img.shape[1]-1)  
  x2 = np.clip(x2,0,img.shape[1]-1)  

  y1 = np.clip(y1,0,img.shape[0]-1)  
  y2 = np.clip(y2,0,img.shape[0]-1)  

```

## 联系方式 （Contact）  
* E-mails: 305141918@qq.com   
