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
	string filename3 = ".\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte";  // ��������
	string filename4 = ".\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte";

	bpData test_data(filename3, filename4, 10000);  // ����������

	bpNet newNet(".\\60000_5matrix.txt");                      // ����ģ��
	cout << "׼ȷ�ʣ�" << test_bp(newNet, test_data) << endl;  // ������Խ��
}