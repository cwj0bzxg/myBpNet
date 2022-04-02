/*
* author: cwj
* date: 2022年4月2日
*/

#include<fstream>
#include<iostream>

#include"BpData.h"
#include"BpNet.h"
using namespace std;
int main()
{
	string filename1 = ".\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte";  // 标签文件
	string filename2 = ".\\train-images-idx3-ubyte\\train-images.idx3-ubyte";  // 图片文件

	bpData train_data(filename1, filename2, 60000);  // 读取 60000 条数据

	int input_num = train_data.get_n_cols() * train_data.get_n_rows();  // 输入神经元数
	int output_num = 10;          // 输出神经元数
	double learningRate = 0.02;   // 学习率

	bpNet myNet(input_num, output_num, learningRate);  // 初始化Bp神经网络类
	myNet.addNeuronNet(input_num, 28);  // 两个隐藏层
	myNet.addNeuronNet(28, 28);
	myNet.addNeuronNet(28, 10);         // 输出层

	int epoch = 5;
	if (train_bp(myNet, train_data,epoch)) {        // 训练模型
		if (myNet.save_model(".\\matrixTest.txt"))  // 保存模型
			cout << "保存成功！" << endl;
		else
			cout << "保存失败..." << endl;
	}
	else
		cout << "error!" << endl;
}