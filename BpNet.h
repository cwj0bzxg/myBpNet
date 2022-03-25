/*
author��cwj
date��2022/03/24-26
���ͷ�ļ�ʵ����һ���򵥵�BP������
*/

#pragma once
#include<vector>
#include<Eigen/Dense>
#include<math.h>
#include<memory>
#include <fstream>
#include<string>
#include<iostream>

class bpNet;  // ����bpNet��

class neuronNet
{
	/* neuronNet�ฺ��洢���������Ȩ�ء�����ƫ�á���Ԫ��ֵ����
	   ���������һ���������±�����Ԫ��ֵ��
	   ������±��������Ȩ�ء�����ƫ�ã�
	   �����򴫲����
	*/
	friend class bpNet;  // ������Ԫ��
public:
	neuronNet(int pnum, int num, double learningRate_=0.01);
	bool forward_bp(Eigen::VectorXd& fInput);  // ��ǰ����
	bool backward_bp(Eigen::VectorXd& preErr,Eigen::VectorXd& preVal);  // ���򴫲�

private:
	int preLayerNum;  // ǰһ�����Ԫ����
	int LayerNum;  // �������Ԫ����

	// Ȩ�ؾ�����preLayerNum�У�LayerNum��
	Eigen::MatrixXd w;  // Ȩ�ؾ���
	Eigen::VectorXd val;  // ��Ԫ��ֵ
	Eigen::VectorXd err;  // ���򴫲����
	Eigen::VectorXd bias;  // ƫ��
	double learningRate; // ѧϰ��
};

class bpNet
{
	/*
	bpNet�ฺ��ά�����neuronNet���󣬼�ά�����������磬
	�������һ����������������BP�������ȫ��������
	�������ѵ���õ��������Ԥ��
	*/
public:
	bpNet(int inputnum, int outputnum,double learningRate_ = 0.01);  // ���캯��

	void addNeuronNet(int pnum, int num);  // ��bp����������һ����Ԫ
	// ѵ��һ������(Input,Taget),������ȫ������
	bool trian_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Taget);  // ��׼BP�㷨������һ����������һ�β���
	// ����ѵ���õ����������Ԥ��
	bool predict_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Output);  //InputΪԤ�����룬OutputΪԤ����

	int getInputNum() { return InputNum; }
	int getOutputNum() { return OutputNum; }

private:
	// ʹ��һ��vector�����洢���ز�������
	std::vector<std::unique_ptr<neuronNet> > bpNeuronNet{};  // ������Ԫ����һ��ֻ��ָ�룬ָ��һ��neuronNet��
	double learingRate;  // ѧϰ��
	int InputNum;  // ��������Ԫ��
	int OutputNum;  // ��������Ԫ��
	Eigen::VectorXd placeholder;  // ռλ�������ù���
};



bpNet::bpNet(int inputnum, int outputnum, double learningRate_)
{
	InputNum = inputnum;
	OutputNum = outputnum;
	learingRate = learningRate_;
	placeholder = Eigen::VectorXd::Random(inputnum); 
}


//����һ�㣨���������ز㣩
void bpNet::addNeuronNet(int pnum, int num)
{
	bpNeuronNet.push_back(std::unique_ptr<neuronNet>(new neuronNet(pnum,num,learingRate)));
}

// ��һ������ѵ������
bool bpNet::trian_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Taget)
{
	int bp_size = bpNeuronNet.size();  // bpNet�洢���������������ʵ�ʵ�BP��������һ�㣬����������㣩

	// ��������������
	for (int index = 0; index < bp_size; index++) {
		bpNeuronNet[index]->err = Eigen::VectorXd::Zero(bpNeuronNet[index]->LayerNum);
	}
	// ��ǰ����
	bpNeuronNet[0]->forward_bp(Input);
	for (int index = 1; index < bp_size; index++) {
		bpNeuronNet[index]->forward_bp(bpNeuronNet[index - 1]->val);
	}
	Eigen::VectorXd Err = Taget - bpNeuronNet[bp_size - 1]->val;  // ���һ���val��Ϊ���

	// ���򴫲�
	bpNeuronNet[bp_size - 1]->err = Err;
	for (int index = bp_size - 1; index > 0; index--) {
		bpNeuronNet[index]->backward_bp(bpNeuronNet[index - 1]->err, bpNeuronNet[index - 1]->val);
	}
	bpNeuronNet[0]->backward_bp(placeholder, Input);

	return true;
}

// Ԥ�⺯��
bool bpNet::predict_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Output)
{
	// ��ǰ����
	bpNeuronNet[0]->forward_bp(Input);
	for (int index = 1; index < bpNeuronNet.size(); index++) {
		bpNeuronNet[index]->forward_bp(bpNeuronNet[index - 1]->val);
	}
	Output = bpNeuronNet[bpNeuronNet.size() - 1]->val;  // ���һ���val��Ϊ���
	return true;
}

neuronNet::neuronNet(int pnum, int num, double learningRate_)
{
	this->preLayerNum = pnum;
	this->LayerNum = num;
	this->learningRate = learningRate_;
	// �����ֵ
	this->w = Eigen::MatrixXd::Random(num,pnum);  // ����Ȩ��[preLayerNum��LayerNum]
	this->val = Eigen::VectorXd::Random(num);  // ������Ԫ��ֵ[LayerNum��1]
	this->bias = Eigen::VectorXd::Random(num);  // ���������ƫ��[LayerNum��1]

	this->err = Eigen::VectorXd::Zero(num);  // �����Ԥ�����[LayerNum��1](���BP�������У���Ҫ���򴫲����)
}

bool neuronNet::forward_bp(Eigen::VectorXd& fInput)
{
	// �жϾ��������Ƿ�������
	if (fInput.rows() != preLayerNum) return false;
	
	//��������Ȩ�ؾ���ƫ�ú�ǰһ����Ԫ��ֵ���㱾����Ԫ��ֵ
	//1.0 / (1.0 + exp(-x))��sigmoid�����
	val = (w * fInput + bias).unaryExpr([](double x) {return 1.0 / (1.0 + exp(-x)); });
	return true;
}

bool neuronNet::backward_bp(Eigen::VectorXd& preErr, Eigen::VectorXd& preVal)
{
	// ȷ����������������
	if (preErr.rows() != preVal.rows()) return false;

	Eigen::VectorXd middle_val,convert_sigmoid_val;  // �м�������������
	convert_sigmoid_val = val.unaryExpr([](double x) {return x * (1 - x); });  //x * (1 - x)��sigmoid������ĵ���|���򴫲������
	auto mid_err_matrix = convert_sigmoid_val.asDiagonal();  // ���ɶԽǾ�����ʵ������������ӦԪ�����
	middle_val = (mid_err_matrix * err).transpose();

	preErr += w.transpose() * err;  // ���򴫵������ݱ������������һ������
	w += middle_val * preVal.transpose()*learningRate;  // ����Ȩ��
	bias -= middle_val*learningRate;  // ����ƫ��bias

	return true;
}

