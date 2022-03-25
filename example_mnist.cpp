#include<Eigen/Dense>
#include<iostream>
#include"BpNet.h"
#include<time.h>
#include"BpMnistDemo.h"
using namespace std;


int main()
{
	clock_t point1, point2, point3, point4, point5, point6;  // 计时
	point1 = clock();

	bpNet myNet(784, 10);  // 初始化一个BP神经网络
	myNet.addNeuronNet(784, 16);  // 插入两个隐藏层
	myNet.addNeuronNet(16, 16);
	myNet.addNeuronNet(16, 10);  // 最后插入一个输出层

	bpMnistData myData;  
	point2 = clock();

	myData.load_train("./mnist_train_500.csv");  // 导入训练数据
	point3 = clock();

	train_mnist(myNet, myData, 5);  // 由于样本较多，只需要训练5次就能得到很好的准确率
	point4 = clock();
	myData.load_test("./mnist_test_100.csv");  // 导入测试数据
	point5 = clock();
	double accuracy = test_mnist(myNet, myData);  // 输出准确率
	point6 = clock();
	cout << "准确率:" << accuracy << endl;

	// 输出每个步骤的耗时
	cout << "初始化：" << point2 - point1 << "ms\n"
		<< "加载训练数据" << point3 - point2 << "ms\n"
		<< "训练模型：" << point4 - point3 << "ms\n"
		<< "加载测试数据:" << point5 - point4 << "ms\n"
		<< "测试：" << point6 - point5 << "ms" << endl;
}