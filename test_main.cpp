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
	string filename3 = ".\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte";  // 测试数据
	string filename4 = ".\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte";

	bpData test_data(filename3, filename4, 10000);  // 测试数据类

	bpNet newNet(".\\60000_5matrix.txt");                      // 载入模型
	cout << "准确率：" << test_bp(newNet, test_data) << endl;  // 输出测试结果
}