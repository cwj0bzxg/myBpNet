### 项目简介
本项目用C++实现了一个轻量的多层BP神经网络（BpNet.h），并使用这个BP神经网络进行手写数字识别（BpMnistDemo.h）  

### 环境
C++11、Eigen

### 使用
直接将BpNet.h或BpMnistDemo.h放到你的项目的头文件夹中即可

### 文件介绍
- BpNet.h  
这个头文件实现了标准BP算法，提供了训练函数和预测函数  
- BpMnistDemo.h  
这个头文件调用BpNet.h头文件封装的反向传播算法实现手写数字识别。文件中定义了一个类bpMnistData，对Mnist数据集进行管理，并定义了三个函数分别进行手写数字识别的训练、测试和预测
- example_bpnet.cpp  
演示头文件BpNet.h的使用
- example_mnist.cpp  
演示头文件example_mnist.cpp的使用
- mnist_train_500.csv，mnist_test_100  
Mnist的训练数据的测试数据  

### 参考资料
1. 《机器学习》 周志华
2. https://blog.csdn.net/xuanwolanxue/article/details/71565934
3. https://eigen.tuxfamily.org/dox/group__QuickRefPage.html
