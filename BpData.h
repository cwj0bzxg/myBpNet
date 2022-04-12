/*
* author: cwj
* data: 2022年4月1日
*/

#pragma once
#include"BpNet.h"
#include<string>
#include<iostream>
#include<fstream>
#include<string>
/* 
* 这个头文件调用BpNet.h头文件封装的反向传播算法实现手写数字识别。
* 这个文件中定义了一个类bpMnistData，对Mnist数据集进行管理，
* 并定义了三个函数分别进行手写数字识别的训练、测试和预测
*/
class bpData;

bool train_bp(bpNet& myNet, bpData& myData, int epoch = 5);  // 训练，默认训练 5 次
double test_bp(bpNet& myNet, bpData& myData);  // 测试
int predict_bp(bpNet& myNet, Eigen::VectorXd& Input);  // 预测

class bpData
{
	// 声明友元函数，以便在train_bp和test_bp中直接调用bpData的私有成员变量
	friend bool train_bp(bpNet& myNet, bpData& myData, int epoch);
	friend double test_bp(bpNet& myNet, bpData& myData);
public:
	// label_filename：标签文件、images_filename：图片文件、max_rows：载入行数
	bpData(const std::string& label_filename, const std::string& images_filename,int max_rows=100000);
	//bpData() = default;
	bool read_label();  // 载入标签数据
	bool read_images(); // 载入图片数据
	int get_number_of_label() { return std::min(_number_of_label, _max_rows); }   // 返回标签数量
	int get_number_of_images() { return std::min(_number_of_images, _max_rows); } // 返回图片数量
	int get_n_rows() { return _n_rows; }  // 返回一张图片有多少行
	int get_n_cols() { return _n_cols; }  // 返回一张图片有多少列
private:
	bool read_label_info(const std::string& filename);  // 获取数据集信息
	bool read_images_info(const std::string& filename);
	Eigen::VectorXi _label_vec;  // 标签数据
	Eigen::MatrixXd _images_mat; // 图片数据
	std::string _label_file;     // 标签文件地址
	std::string _images_file;    // 图片文件地址
	int _number_of_label;        // 标签数量
	int _number_of_images;       // 图片数量
	int _n_rows;                 // 一张图片的行数
	int _n_cols;                 // 一张图片的行数
	int _max_rows;               // 读入行数
};

int reverse_int(int i);          // 对 i 进行一些位运算
