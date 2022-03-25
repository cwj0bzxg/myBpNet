#include<Eigen/Dense>
#include<iostream>
#include"BpNet.h"
#include<time.h>
using namespace std;


int main()
{
	// ������������
	Eigen::VectorXd vInput(10), vInput1(10),vInput2(10);
	Eigen::VectorXd vOuput(4);
	Eigen::VectorXd Taget(4), Taget1(4),Taget2(4);
	vInput << 0, 0, 0, 0, 0, 1, 0, 0, 0, 0;
	vInput1 << 0, 1, 0, 0, 0, 0, 0, 0, 0, 0;
	vInput2<< 0, 0, 0, 0, 0, 0, 0, 0, 1, 0;
	Taget << 0, 1, 0, 1;
	Taget1 << 1, 0, 1, 0;
	Taget2 << 1, 0, 0, 1;

	clock_t start, end;

	bpNet layerNet(10, 4);  // ��ʼ��BP������
	layerNet.addNeuronNet(10, 16);  // �����������ز�
	layerNet.addNeuronNet(16, 16);  
	layerNet.addNeuronNet(16, 4);  // ����һ�������
	start = clock();
	for (int i = 0; i < 2000; i++) {  // ѭ��2000��
		// ���θ����������������ѵ��������
		layerNet.trian_bp(vInput, Taget);
		layerNet.trian_bp(vInput1, Taget1);
		layerNet.trian_bp(vInput2, Taget2);
	}
		
	end = clock();

	// ���Ԥ����
	cout << "--0101--" << endl;
	layerNet.predict_bp(vInput, vOuput);
	cout << vOuput << endl;

	cout << "\n--1010--" << endl;
	layerNet.predict_bp(vInput1, vOuput);
	cout << vOuput << endl;

	cout << "\n--1001--" << endl;
	layerNet.predict_bp(vInput2, vOuput);
	cout << vOuput << endl;

	cout << "time = " << double(end - start) << "ms" << endl;
}