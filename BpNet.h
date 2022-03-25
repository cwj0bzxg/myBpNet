/*
author：cwj
date：2022/03/24-26
这个头文件实现了一个简单的BP神经网络
*/

#pragma once
#include<vector>
#include<Eigen/Dense>
#include<math.h>
#include<memory>
#include <fstream>
#include<string>
#include<iostream>

class bpNet;  // 声明bpNet类

class neuronNet
{
	/* neuronNet类负责存储本层的输入权重、输入偏置、神经元的值和误差，
	   负责根据上一层的输入更新本层神经元的值，
	   负责更新本层的输入权重、输入偏置，
	   负责反向传播误差
	*/
	friend class bpNet;  // 声明友元类
public:
	neuronNet(int pnum, int num, double learningRate_=0.01);
	bool forward_bp(Eigen::VectorXd& fInput);  // 向前传播
	bool backward_bp(Eigen::VectorXd& preErr,Eigen::VectorXd& preVal);  // 反向传播

private:
	int preLayerNum;  // 前一层的神经元数量
	int LayerNum;  // 本层的神经元数量

	// 权重矩阵有preLayerNum行，LayerNum列
	Eigen::MatrixXd w;  // 权重矩阵
	Eigen::VectorXd val;  // 神经元的值
	Eigen::VectorXd err;  // 反向传播误差
	Eigen::VectorXd bias;  // 偏置
	double learningRate; // 学习率
};

class bpNet
{
	/*
	bpNet类负责维护多个neuronNet对象，即维护整个神经网络，
	负责根据一个样本来更新整个BP神经网络的全部参数，
	负责根据训练好的网络进行预测
	*/
public:
	bpNet(int inputnum, int outputnum,double learningRate_ = 0.01);  // 构造函数

	void addNeuronNet(int pnum, int num);  // 向bp神经网络增加一层神经元
	// 训练一个样本(Input,Taget),并更新全部参数
	bool trian_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Taget);  // 标准BP算法，传入一个样本更新一次参数
	// 根据训练好的神经网络进行预测
	bool predict_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Output);  //Input为预测输入，Output为预测结果

	int getInputNum() { return InputNum; }
	int getOutputNum() { return OutputNum; }

private:
	// 使用一个vector容器存储隐藏层和输出层
	std::vector<std::unique_ptr<neuronNet> > bpNeuronNet{};  // 容器的元素是一个只能指针，指向一个neuronNet类
	double learingRate;  // 学习率
	int InputNum;  // 输入层的神经元数
	int OutputNum;  // 输出层的神经元数
	Eigen::VectorXd placeholder;  // 占位符，不用管它
};



bpNet::bpNet(int inputnum, int outputnum, double learningRate_)
{
	InputNum = inputnum;
	OutputNum = outputnum;
	learingRate = learningRate_;
	placeholder = Eigen::VectorXd::Random(inputnum); 
}


//增加一层（输出层或隐藏层）
void bpNet::addNeuronNet(int pnum, int num)
{
	bpNeuronNet.push_back(std::unique_ptr<neuronNet>(new neuronNet(pnum,num,learingRate)));
}

// 用一个样本训练网络
bool bpNet::trian_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Taget)
{
	int bp_size = bpNeuronNet.size();  // bpNet存储的神经网络层数（比实际的BP神经网络少一层，不包含输入层）

	// 将各层的误差置零
	for (int index = 0; index < bp_size; index++) {
		bpNeuronNet[index]->err = Eigen::VectorXd::Zero(bpNeuronNet[index]->LayerNum);
	}
	// 向前传播
	bpNeuronNet[0]->forward_bp(Input);
	for (int index = 1; index < bp_size; index++) {
		bpNeuronNet[index]->forward_bp(bpNeuronNet[index - 1]->val);
	}
	Eigen::VectorXd Err = Taget - bpNeuronNet[bp_size - 1]->val;  // 最后一层的val即为输出

	// 反向传播
	bpNeuronNet[bp_size - 1]->err = Err;
	for (int index = bp_size - 1; index > 0; index--) {
		bpNeuronNet[index]->backward_bp(bpNeuronNet[index - 1]->err, bpNeuronNet[index - 1]->val);
	}
	bpNeuronNet[0]->backward_bp(placeholder, Input);

	return true;
}

// 预测函数
bool bpNet::predict_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Output)
{
	// 向前传播
	bpNeuronNet[0]->forward_bp(Input);
	for (int index = 1; index < bpNeuronNet.size(); index++) {
		bpNeuronNet[index]->forward_bp(bpNeuronNet[index - 1]->val);
	}
	Output = bpNeuronNet[bpNeuronNet.size() - 1]->val;  // 最后一层的val即为输出
	return true;
}

neuronNet::neuronNet(int pnum, int num, double learningRate_)
{
	this->preLayerNum = pnum;
	this->LayerNum = num;
	this->learningRate = learningRate_;
	// 赋随机值
	this->w = Eigen::MatrixXd::Random(num,pnum);  // 输入权重[preLayerNum×LayerNum]
	this->val = Eigen::VectorXd::Random(num);  // 本层神经元的值[LayerNum×1]
	this->bias = Eigen::VectorXd::Random(num);  // 本层的输入偏置[LayerNum×1]

	this->err = Eigen::VectorXd::Zero(num);  // 本层的预测误差[LayerNum×1](多层BP神经网络中，需要反向传播误差)
}

bool neuronNet::forward_bp(Eigen::VectorXd& fInput)
{
	// 判断矩阵运算是否有意义
	if (fInput.rows() != preLayerNum) return false;
	
	//根据输入权重矩阵、偏置和前一层神经元的值计算本层神经元的值
	//1.0 / (1.0 + exp(-x))：sigmoid激活函数
	val = (w * fInput + bias).unaryExpr([](double x) {return 1.0 / (1.0 + exp(-x)); });
	return true;
}

bool neuronNet::backward_bp(Eigen::VectorXd& preErr, Eigen::VectorXd& preVal)
{
	// 确保矩阵运算有意义
	if (preErr.rows() != preVal.rows()) return false;

	Eigen::VectorXd middle_val,convert_sigmoid_val;  // 中间变量，便于理解
	convert_sigmoid_val = val.unaryExpr([](double x) {return x * (1 - x); });  //x * (1 - x)：sigmoid激活函数的导数|反向传播激活函数
	auto mid_err_matrix = convert_sigmoid_val.asDiagonal();  // 生成对角矩阵，以实现两个向量对应元素相乘
	middle_val = (mid_err_matrix * err).transpose();

	preErr += w.transpose() * err;  // 反向传递误差，根据本层误差计算出上一层的误差
	w += middle_val * preVal.transpose()*learningRate;  // 更新权重
	bias -= middle_val*learningRate;  // 更新偏置bias

	return true;
}

