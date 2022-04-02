/*
author��cwj
date��2022/03/24-26
���ͷ�ļ�ʵ����һ���򵥵�BP������
*/

#pragma once
#include<vector>
#include<Eigen/Dense>
#include<math.h>
#include<fstream>

class bpNet;  // ����bpNet��
class neuronNet;

class neuronNet
{
	/* neuronNet�ฺ��洢���������Ȩ�ء�����ƫ�á���Ԫ��ֵ����
	   ���������һ���������±�����Ԫ��ֵ��
	   ������±��������Ȩ�ء�����ƫ�ã�
	   �����򴫲����
	*/
	friend class bpNet;  // ������Ԫ��
public:

	bool forward_bp(Eigen::VectorXd& fInput);  // ��ǰ����
	bool backward_bp(Eigen::VectorXd& preErr, Eigen::VectorXd& preVal);  // ���򴫲�

	int getPreLayerNum() { return preLayerNum; }
	int getLayerNum() { return LayerNum; }

private:
	neuronNet(int pnum, int num, double learningRate_);

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
	bpNet(int inputnum, int outputnum, double learningRate_ = 0.05);  // ���캯��
	bpNet(const std::string& filename);

	void addNeuronNet(int pnum, int num);  // ��bp����������һ����Ԫ
	// ѵ��һ������(Input,Taget),������ȫ������
	bool trian_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Taget);  // ��׼BP�㷨������һ����������һ�β���
	// ����ѵ���õ����������Ԥ��
	bool predict_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Output);  //InputΪԤ�����룬OutputΪԤ����

	int getInputNum() { return InputNum; }
	int getOutputNum() { return OutputNum; }

	double getLearningRate() { return learingRate; }

	bool save_model(const std::string& filename);  // ����ģ��
	bool load_model(const std::string& filename);  // ����ģ��
private:
	// ʹ��һ��vector�����洢���ز�������
	std::vector<neuronNet> bpNeuronNet;  // ������Ԫ����һ��ֻ��ָ�룬ָ��һ��neuronNet��
	double learingRate;  // ѧϰ��
	int InputNum;  // ��������Ԫ��
	int OutputNum;  // ��������Ԫ��
	Eigen::VectorXd placeholder;  // ռλ�������ù���
};
