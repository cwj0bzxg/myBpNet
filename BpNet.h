/*
author：cwj
date：2022/03/24-26
这个头文件实现了一个简单的BP神经网络
*/

#pragma once
#include<vector>
#include<Eigen/Dense>
#include<math.h>
#include<fstream>

class bpNet;  // 声明bpNet类
class neuronNet;

class neuronNet
{
	/* neuronNet类负责存储本层的输入权重、输入偏置、神经元的值和误差，
	   负责根据上一层的输入更新本层神经元的值，
	   负责更新本层的输入权重、输入偏置，
	   负责反向传播误差
	*/
	friend class bpNet;  // 声明友元类
public:

	bool forward_bp(Eigen::VectorXd& fInput);  // 向前传播
	bool backward_bp(Eigen::VectorXd& preErr, Eigen::VectorXd& preVal);  // 反向传播

	int getPreLayerNum() { return preLayerNum; }
	int getLayerNum() { return LayerNum; }

private:
	neuronNet(int pnum, int num, double learningRate_);

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
	bpNet(int inputnum, int outputnum, double learningRate_ = 0.05);  // 构造函数
	bpNet(const std::string& filename);

	void addNeuronNet(int pnum, int num);  // 向bp神经网络增加一层神经元
	// 训练一个样本(Input,Taget),并更新全部参数
	bool trian_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Taget);  // 标准BP算法，传入一个样本更新一次参数
	// 根据训练好的神经网络进行预测
	bool predict_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Output);  //Input为预测输入，Output为预测结果

	int getInputNum() { return InputNum; }
	int getOutputNum() { return OutputNum; }

	double getLearningRate() { return learingRate; }

	bool save_model(const std::string& filename);  // 保存模型
	bool load_model(const std::string& filename);  // 载入模型
private:
	// 使用一个vector容器存储隐藏层和输出层
	std::vector<neuronNet> bpNeuronNet;  // 容器的元素是一个只能指针，指向一个neuronNet类
	double learingRate;  // 学习率
	int InputNum;  // 输入层的神经元数
	int OutputNum;  // 输出层的神经元数
	Eigen::VectorXd placeholder;  // 占位符，不用管它
};
