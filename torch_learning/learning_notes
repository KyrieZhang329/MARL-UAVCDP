# B站土堆PyTorch课程学习笔记

*这个是我在跟随B站土堆老师学习Torch时所记的笔记，某些地方的图片和代码借鉴了csdn上Cheer-ego的学习笔记，在此致谢，本篇笔记较为简略，若有不懂之处可以前往csdn上的详细笔记进行学习*，原文转载声明如下：

*版权声明：本文为CSDN博主「Cheer-ego」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_43629945/article/details/122767670*



[TOC]

## Pytorch(or python)学习时的两大工具函数

### dir( )函数--打开package

传入一个package，可以输出该package中包含的分隔区，如果分隔区中依旧为package，则可以继续调用dir（）打开，直到显示“__ 函数名 __”，则代表上一步打开的是一个可以调用的函数

e.g.

``` python
import torch
dir(torch)
dir(torch.cuda)
dir(torch.cuda.is_available)
```

### help( )函数--查询函数的用法

传入一个函数（没有括号） 得到官方对函数用法的介绍

在PyCharm中按住ctrl再点击该函数或类也可以找到说明

e.g.

```python
import torch
help(torch.cuda.is_available)
```





## torch中加载数据

torch中加载数据一般涉及到两个类：**Dataset** & **Dataloader(将会在后面介绍)**

### 数据集组织形式

- 文件夹名为label
- 文件名为label
- 两个文件夹中 一个为图片 一个为txt文档储存label 一一对应

### 获取图片

#### PIL获取

返回一个

```python
from PIL import Image
img = Image.open("图片路径")
# windows系统中需要将路径改为双斜杠避免识别为转义符
```

#### opencv获取

返回一个numpy数组

```python
import cv2
img = cv2.imread("图片路径")
```

### Dataset 

Dataset的引入：

```python
from torch.utils.data import Dataset
```

作用：**获取数据及数据的label**

所有的数据集都需要来继承Dataset类

Dataset的继承和使用

```python
from PIL import Image
import os # 加载数据文件夹(获取图片地址列表)

class Mydata(Dataset):

	def __init__(self，root_dir,label_dir):
        # root_dir = "文件夹地址"
        # label_dir = "label文件夹地址（标注label）"
        # path = os.path.join(root_dir,label_dir) # 合成拼接文件地址
        self.root_dir = root_dir
		self.label_dir = label_dir
        self.img_path_list = os.listdir(os.path.join(self.root_dir,self.label_dir))
        # 获取图片地址列表
        
	def __getitem__(self,index): # index表示下标
		img_name = self.img_path[index]
        img_item_path = os.path.join(self.img_path_list,img_name)
        img = Image.open.(img_item_path)
        label = self.label_dir
        return img,label
    
    def __len__(self):
        return len(self.img_path_list)
    
```





## Tensorboard的使用

Tensorboard的引用和主要涉及类SummaryWriter

```python
from Tensorboard import SummaryWriter
writer = SummaryWriter("文件夹")

# 主要涉及两个方法函数
writer.add_image( )
writer.add_scalar( )

writer.close( )
```

### add_scalar( )方法函数的使用

```python
add_scalar(tag(图表的title), value(y轴)，step(x轴) )
```

#### 打开Tensorboard事件文件

在terminal中输入tenseorboard --logdir=*"文件名"*  （--port="设置端口"）

每次写入新事件时，需要改变子文件名

### add_image方法函数的使用

```python
add_image( tag，img_tensor，step )
```

在传入img_tensor时，要注意数据的shape，default为`（3, H, W）`（通道，高度，宽度）

通过`.shape`方法来查看数据shape，若与默认不同，则需在传入变量时后加入dataformats= '数据的shape'（HW3）





## Transforms的使用

### Transforms的引入和结构用法

```python
from torchvision import transforms
```

主要涉及类：**ToTensor**、**resize**

### Tensor数据类型

n阶tensor就是特殊的n维数组，包装了神经网络需要的参数，在神经网络深度学习中使用广泛



### 常见的Transforms

#### ToTensor

将numpy数组(opencv) 和PIL(PIL) 图片格式转化为Tensor数据类型 

```python
tensor_trans = transforms.ToTensor( )
tensor_img = tensor_trans("图片") # 按ctrl+P可以查看需要传入的数据类型
# result = tool(input)
```

#### Compose

将不同的transforms类结合在一起发挥作用

```python
transforms_compose = transforms.Compose( )
transforms_Compose = transformcompse([transfors.ToTensor(),transforms.CenterCrop(10)])
# Compose中需要传入一个列表，列表元素均为transforms，后续可以利用创建的compose类来同时依次调用列表里所有的transforms，前一个的return必须与后一个的输入为相同的数据类型
```

#### ToPILImage

与ToTensor用法相同，可以将Tensor转化为PILimage数据类型

#### Nomalize

传入平均值和标准差来归一化Tensor，需输入与信道数相同数量的数，Image为RAG三信道，故传入三个值

```python
trans_norm = transforms.Nomalize([mean1,mean2,mean3],[std1,std2,std3])
img_norm = trans_norm("Tensor类型的图片")
# 计算公式：output[channel] = (input[channel] - mean[channel])/ std[channel]
```

#### Resize

若传入高度和宽度（h，w），则会将图片调整为对应高度和宽度

若只传入一个数，则会进行等比缩放，较小的一个边大小改变为传入的参数，另一个边与其比例不变改变

```python
trans_resize = transforms.Resize(h, w)
img_resize = trans_resize("PIL类型的图片")
# 一般来说 会使用Compose将Resize和ToTensor结合起来一起使用
trans_totensor = transforms.ToTensor()
trans_compose = transforms.Compose([trans_resize,trans_totensor]) 
# 传入的参数是前一个transforms的参数
img_2 = trans_compose(img)
```

#### RandomCrop

设置类传入可以传入一个参数作为指定的宽和高，一个参数作为宽高，可以随机裁剪指定大小的图片

```python
trans_random = transforms.RandomCrop(512，1024)
# 一般也结合ToTensor使用
trans_totensor = transforms.ToTensor()
trans_compose = tansforms.Compose([trans_random,trans_totensor])
for i in range(10)
	img_crop = trans_compose(img)
    writer.add_image("RandomCrop",img,crop,i)
```

#### 总结transforms的使用

- 关注输入和输出类型

- 多看官方文档了解用法

- 关注方法需要的参数！ *print(type( ))*

  



## torchvision中数据集的使用

#### torchvision.datasets

提供的数据集，常用包括：COCO目标检测、语义分割；MNIST手写文字；CIFAR物体识别

#### Torchvision.models

提供一些已训练好的常见的神经网络

### torchvision.datasets数据集和transforms的连用

根据官方文档数据集的说明传入参数![img](https://i-blog.csdnimg.cn/blog_migrate/d13320bae2a96b72a594b5dac7d5cca1.png)

```python
import torchvision

dataset_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 创建的均为列表类
train_set = torchvision.datasets.CIFAR10(root ="文件数据集地址"，train = True, transform = dataset_transforms,download = True)
test_set = torchvision.datasets.CIFAR10(root ="文件数据集地址"，train = False, transform = dataset_transforms,download = False)
```





## Dataloader的使用

Dataloader传入的参数（常见）：

- **dataset**： 需要操作的数据集
- **batch_size**： 每一步从数据集中取出的数据
- **shuffle**： 是否打乱取数据 （bool值）
- **num_workers**： 多进程/单进程控制 一般设置为0
- **drop_last**： 最后一步取出的数据若不够batch_size设置的值，是否依旧取出不够的数据（bool值）

```python
import torchvision

test_data = torchvision.datasets.CIFAR10(root ="文件数据集地址"，train = False, transform = dataset_transforms,download = False)
# 为一个列表，输入对应下标返回值为对应的img和target

tset_loader = DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0,drop_last=False)

img,target = test_data[0]

for data in test_loader:
    imgs,targets = data
# DataLoader的返回值是将每次取出的imgs和targets打包给出
	
```



## 神经网络基本骨架的搭建-nn.Module的使用

  主要涉及API： ***torch.nn***、***torch.nn.functional***

搭建神经网络时，以nn.Module为父类框架进行子类继承搭建

```python
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
	def __init__(self):
        super(Model,self).__init__() # 继承父类
        """
        子类自己搭建神经网络时所添加的内容
        """
        
	def forward(self,x): # 神经网络操作计算（输入-输出）
        """
        神经网络的操作计算
        """
```



## 卷积操作

**torch.nn** 与 **torch.nn.functional** 区别：前者为后者的封装，torch.nn可以直接使用，torch.nn.functional是具体函数方法

主要涉及方法：**nn.Conv1d**(一维操作)、**nn.Conv2d**（二维操作）、**nn.Conv3d**(三维操作)

e.g. - nn.Conv2d

需传入参数：

- **input**: Tensor（数据尺寸标注为（batch, channel, H, W）,若维度不匹配时，可以使用torch.reshape（）方法来进行变换）
- **weight**: 卷积核/权重，同样为Tensor数据类型，对input进行卷积的标准工具
- **stride**：步进，表示每次操作时weight在input上移动的步长
- **padding**：default为0，设定为int数据类型，会在input外为分别填充输入的参数的行列数，在填充后的input上再进行卷积操作

```python
import torch
import torch.nn.functional as F

inp = torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]])

kernel =  torch.tensor([[1,2,1],
                        [0,1,0],
                        [2,1,0]])

inp = torch.reshape(inp,(1,1,5,5)) # 修改时，某个参数填-1 该参数会根据后面的参数自动计算
kernel = torch.reshape(kernel,(1,1,3,3))

output = F.conv2d(inp,kernel,stride=1,padding=1)
```



## 卷积层

图片为2d矩阵，故以conv2d为示例，具体用法可查询官网

传入参数：

- **in_channels**: 输入图片的通道数（几层，一般为3）
- **out_channels**: 输出图片的通道数，设定为n时，卷积层生产n个卷积核进行卷积得到n个输出叠加在一起
- **kernel_size**: 卷积核的大小尺寸，设定后从分布采样中随时调整设定具体卷积核
- **stride**: 步进，表示每次操作时卷积核在input上移动的步长
- **padding**:default为0，设定为int数据类型，会在input外为分别填充输入的参数的行列数，在填充后的input上再进行卷积操作

```python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
 
dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=True) 
dataloader = DataLoader(dataset,batch_size=64)
 
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,stride=1,padding=0)   
   
	def forward(self,x): 
        x = self.conv1(x)
        return x
 
tudui = Tudui()  
 
writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs,targets = data  
    output = tudui(imgs)
    print(imgs.shape)    
    print(output.shape)   
    writer.add_images("input",imgs,step)
   
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("output",output,step)
    step = step + 1
```





## 最大池化层

作用：压缩数据文件尺寸，提高训练效率

传入参数：

- **input**

- **kernel_size**
- **stride**： 默认为kernel_size
- **padding**

- **dilation**：空洞，即在Tensor中每个单位数据之间缺空
- **ceil_mode**:设置为True时，采用ceil模式，即在最后存在余数，保留最后一次池化操作得到的值；设置为False，采用floor模式，不保留最后一次存在余数的操作值 （默认为false）

```python
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
 
inp = torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]],dtype=torch.float32) # 设置数据类型

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool = MaxPool2d(kernel_size=3,ceil_mode=True)
  
	def forward(self,x): 
        x = self.maxpool(inp)
        return x
    
tudui = Tudui()
output = tudui(inp)
```



## 非线性变换层

作用：在神经网络中引入非线性特征，训练出更多符合特征的模型

传入参数

- input

```python
import torch
from torch import nn

inp = torch.tensor([[1,-0.5],
                    [-1,3]])

inp = torch.reshape(inp,(-1,1,2,2))

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu = ReLU()
  
	def forward(self,x): 
        x = self.relu(inp)
        return x
    
tudui = Tudui()
output = tudui(inp)
```



## 正则化层

作用：将数据正则化，提高训练效率

传入参数：

- **num_features**： 由传入数据的通道数决定



## Recurrent层

多用于文字识别



## Transformer层

特定网络中使用



## 线性层

作用：线性转化（k*x+b）传入数据

传入数据：

- input_feature
- output_feature

- bias(bool): 决定是否在转化时加b

```python
# torch.flatten() - 摊平Tensor，将Tensor数据变为一维

import torchvision
import torch
from torch.nn import Linear

class Tudui(nn.Module)
	def __init__(self)
    	super(Tudui,self).__init__()
        self.linear1 = Linear("input参数"，"output参数")
        
    def forward(self,x)
    	output = self.linear1(x)
        return x
```

### Sequential的用法

```python
self.conv1 = nn.Conv2d(3, 32, 5,padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5,padding=2)
        self.maxpool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5,padding=2)
        self.maxpool3 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 10)

        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
        
        # 上下等价
```



## 损失函数与反向传播

作用：

1. 计算实际输出和目标之间的差距
2. 为更新输出提供一定一句（反向传播）

###  L1Loss

计算每一个数据的input和target差值的绝对值

传入参数：

- **input**（N, *）： 只要求有batch_size，float类型
- **target**
- **reduction**: 设置为*mean*时取差值平均值，设置为*sum*时求和

### MSELoss

计算每一个数据的input和target差值的平方

传入参数：

- **input**
- **target**
- **reduction**

```python
import torch
from torch.nn import L1Loss
 

inputs = torch.tensor([1,2,3],dtype=torch.float32)  
targets = torch.tensor([1,2,5],dtype=torch.float32)
 
inputs = torch.reshape(inputs,(1,1,1,3))  
targets = torch.reshape(targets,(1,1,1,3))
 
loss = L1Loss()
result = loss(inputs,targets)

loss_mse = MSELoss()
result_mse = loss_mse(inputs,targets)
```

### CrossEntropyLoss

训练分类问题时常用

![img](https://i-blog.csdnimg.cn/blog_migrate/5e817733bc16eb6c5ef8100e3fd43646.png)

```python
x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x,y)
```

在卷积神经网络中应用loss函数

```python
dataset = torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset,batch_size=1)

class model(nn.Module):
	def __init__(self):
        super(model,self).__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
        
        def forward(self,x):
            x = self.model1(x)
            return x
        
loss = nn.CrossEntropyLoss()

_model = model()

for data in dataloader：
	imgs,targets = data
    outputs = _model(imgs)
    result_loss = loss(outputs,tragets)
    result_loss.backward()  # backward反向传播，是对result_loss，而不是对loss 
    # 只有加上backward才可以反向传播调节参数
```



## 优化器

构造不同算法优化器时必须传入参数：**model.parameters()**，根据需要可以设置学习速率，推荐为0.01

example：

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam([var1, var2], lr=0.0001)
```

构造优化器后，调用优化器step方法来进行参数优化

```python
for epoch in range(20):
    for data in dataloader: # 内层循环只代表将每一个数据优化一轮，需要在外层嵌套一层循环设置学习量
        imgs,targets = data
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
```



## 完整网络模型的训练流程

- 准备数据集（训练集、测试集）
- 准备分别对应的dataloader，用**len()**表示出训练集和测试集的大小
- 定义和创建网络模型（通常单独采用一个文件定义存放）
- 损失函数和优化器创建
- 设置训练参数
  - 训练次数、测试次数
  - 训练轮数

- 设置tensorboard
- 进入循环，开始训练，设置训练状态：

```python
for i in range(epoch):
	model.train()
    for data,targets in train_loader:
        outputs = model(data)
        loss = loss_fn(outputs,targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_steps += 1
        # 每100次记录一次
        if total_train_steps % 100 == 0:
            print(f"训练次数:{total_train_steps},loss:{loss.item()}")
            writer.add_scalar("train_loss",loss.item(),total_train_steps)
```

- 每一轮循环结束，开始测试一次

```python
model.eval()
toatl_test_loss = 0
total_test_accuracy = 0
with torch.no_grad():
    for data,targets in test_loader:
        outputs = model(data)
        loss = loss_fn(outputs,targets)
        total_test_loss += loss.item
        accuracy =(outputs.argmax(1) == targets).sum()
        total_test_accuracy += accuracy

print(f"整体测试集上的loss:{total_test_loss}")
print(f"整体测试及上的accuracy:{total_test_accuracy}")
writer.add_scalar("test_loss",total_test_loss,total_test_step)
writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_steps)
total_test_steps += 1

# 训练完之后记得保存模型
torch.save(model.state_dict(), f"./model{i}.pth")
print("模型已保存")
```

- 关闭tensorboard

```python
writer.close()
```



## 利用GPU进行训练

### 第一种方法

在网络模型、数据（输入、标注）、损失函数分别调用**.cuda**

```python
# 网络模型创建处
model = Net()
model = model.cuda()

# 损失函数处
loss_fn = loss_fn.cuda

#训练和测试分别调用
for data,targets in train_loader:
    data = data.cuda()
    targets = targets.cuda()
    
# 建议在所有调用cuda时，前面判断gpu是否可用
if torch.cuda.is_available():
	
```

### 第二种方法

在开始定义一个设备，在网络模型、数据（输入、标注）、损失函数分别调用**.to(device)**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 网络模型创建处
model = Net()
model = model.to(device)

# 损失函数处
loss_fn = loss_fn.to(device)

#训练和测试分别调用
for data,targets in train_loader:
    data = data.to(device)
    targets = targets.to(device)
```



## 完整的模型测试流程

利用训练好的模型，自行传入参数进行验证（以图片识别为例）

- 用PIL中Image.open()打开传入的图片
- 调用**image.convert('RGB')**将不同的图片类型转化为rgb三通道类型
- 连接需要使用的transforms（Resize：将图片大小改为导入参数、ToTensor：图片数据类型改为tensor）
- 加载训练好的模型，调用**torch.load("模型路径")**方法
- 调用**torch.reshape()**，将Tensor类型改为能传入的类型
- 设置模型为测试模式，*with torch.no_grad()*经过模型训练 
- 打印模型识别结果
- 在不同设备（gpu、cpu）上训练的模型，需要在加载模型时映射到此设备**torch.load("模型路径"，map_location=torch.device('cpu'))**

```python
import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from Net import *
 
image_path = "../imgs/airplane.png" 
image = Image.open(image_path)  
 
image = image.convert('RGB')

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), 
                                            torchvision.transforms.ToTensor()])  

image = transform(image)


model = torch.load("tudui_29_gpu.pth",map_location=torch.device('cpu'))  
image = torch.reshape(image,(1,3,32,32))
model.eval()  
with torch.no_grad():
    output = model(image)
print(output.argmax(1))
```

