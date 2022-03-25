#include<Eigen/Dense>
#include<iostream>
#include"BpNet.h"
#include<time.h>
#include"BpMnistDemo.h"
using namespace std;


int main()
{
	clock_t point1, point2, point3, point4, point5, point6;  // ��ʱ
	point1 = clock();

	bpNet myNet(784, 10);  // ��ʼ��һ��BP������
	myNet.addNeuronNet(784, 16);  // �����������ز�
	myNet.addNeuronNet(16, 16);
	myNet.addNeuronNet(16, 10);  // ������һ�������

	bpMnistData myData;  
	point2 = clock();

	myData.load_train("./mnist_train_500.csv");  // ����ѵ������
	point3 = clock();

	train_mnist(myNet, myData, 5);  // ���������϶ֻ࣬��Ҫѵ��5�ξ��ܵõ��ܺõ�׼ȷ��
	point4 = clock();
	myData.load_test("./mnist_test_100.csv");  // �����������
	point5 = clock();
	double accuracy = test_mnist(myNet, myData);  // ���׼ȷ��
	point6 = clock();
	cout << "׼ȷ��:" << accuracy << endl;

	// ���ÿ������ĺ�ʱ
	cout << "��ʼ����" << point2 - point1 << "ms\n"
		<< "����ѵ������" << point3 - point2 << "ms\n"
		<< "ѵ��ģ�ͣ�" << point4 - point3 << "ms\n"
		<< "���ز�������:" << point5 - point4 << "ms\n"
		<< "���ԣ�" << point6 - point5 << "ms" << endl;
}