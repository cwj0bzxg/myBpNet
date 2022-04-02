### 项目简介
本项目用C++实现了一个轻量的多层BP神经网络，并使用这个BP神经网络进行手写数字识别

### 环境
C++11、Eigen

### 使用
1、下载Eigen库 https://eigen.tuxfamily.org/dox/group__QuickRefPage.html  
2、下载数据集 http://yann.lecun.com/exdb/mnist/  
3、将头文件和Cpp文件复制到你的项目中  
4、修改数据集的文件路径  
5、运行  

### 文件介绍
- BpNet.h、BpNet.cpp  
这两个文件实现了Bp神经网络和反向传播算法  
- BpData.h、BpData.cpp  
文件中定义了一个类对Mnist数据集进行管理，并定义了三个函数分别进行手写数字识别的训练、测试和预测
- example_bpnet.cpp  
头文件BpNet.h封装的函数的用例
- train_main.cpp  
手写数字识别：训练Bp神经网络、保存模型
- test_main.cpp  
手写数字识别：载入Bp神经网络、测试模型
- 60000_5matrix.txt、60000_1matrix.txt  
训练好的模型，准确率分别为0.92和0.89

### 参考资料
1. 《机器学习》 周志华
2. https://blog.csdn.net/xuanwolanxue/article/details/71565934
3. https://eigen.tuxfamily.org/dox/group__QuickRefPage.html
4. https://n3verl4nd.blog.csdn.net/article/details/53052963
5. https://blog.csdn.net/qq_20936739/article/details/82011320
6. https://blog.csdn.net/weixin_41661099/article/details/105223866
