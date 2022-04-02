/*
* author: cwj
* date: 2022��4��2��
*/

#include<fstream>
#include<iostream>

#include"BpData.h"
#include"BpNet.h"
using namespace std;
int main()
{
	string filename1 = ".\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte";  // ��ǩ�ļ�
	string filename2 = ".\\train-images-idx3-ubyte\\train-images.idx3-ubyte";  // ͼƬ�ļ�

	bpData train_data(filename1, filename2, 60000);  // ��ȡ 60000 ������

	int input_num = train_data.get_n_cols() * train_data.get_n_rows();  // ������Ԫ��
	int output_num = 10;          // �����Ԫ��
	double learningRate = 0.02;   // ѧϰ��

	bpNet myNet(input_num, output_num, learningRate);  // ��ʼ��Bp��������
	myNet.addNeuronNet(input_num, 28);  // �������ز�
	myNet.addNeuronNet(28, 28);
	myNet.addNeuronNet(28, 10);         // �����

	int epoch = 5;
	if (train_bp(myNet, train_data,epoch)) {        // ѵ��ģ��
		if (myNet.save_model(".\\matrixTest.txt"))  // ����ģ��
			cout << "����ɹ���" << endl;
		else
			cout << "����ʧ��..." << endl;
	}
	else
		cout << "error!" << endl;
}