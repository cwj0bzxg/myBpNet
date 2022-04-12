/*
* author: cwj
* data: 2022��4��1��
*/

#pragma once
#include"BpNet.h"
#include<string>
#include<iostream>
#include<fstream>
#include<string>
/* 
* ���ͷ�ļ�����BpNet.hͷ�ļ���װ�ķ��򴫲��㷨ʵ����д����ʶ��
* ����ļ��ж�����һ����bpMnistData����Mnist���ݼ����й���
* �����������������ֱ������д����ʶ���ѵ�������Ժ�Ԥ��
*/
class bpData;

bool train_bp(bpNet& myNet, bpData& myData, int epoch = 5);  // ѵ����Ĭ��ѵ�� 5 ��
double test_bp(bpNet& myNet, bpData& myData);  // ����
int predict_bp(bpNet& myNet, Eigen::VectorXd& Input);  // Ԥ��

class bpData
{
	// ������Ԫ�������Ա���train_bp��test_bp��ֱ�ӵ���bpData��˽�г�Ա����
	friend bool train_bp(bpNet& myNet, bpData& myData, int epoch);
	friend double test_bp(bpNet& myNet, bpData& myData);
public:
	// label_filename����ǩ�ļ���images_filename��ͼƬ�ļ���max_rows����������
	bpData(const std::string& label_filename, const std::string& images_filename,int max_rows=100000);
	//bpData() = default;
	bool read_label();  // �����ǩ����
	bool read_images(); // ����ͼƬ����
	int get_number_of_label() { return std::min(_number_of_label, _max_rows); }   // ���ر�ǩ����
	int get_number_of_images() { return std::min(_number_of_images, _max_rows); } // ����ͼƬ����
	int get_n_rows() { return _n_rows; }  // ����һ��ͼƬ�ж�����
	int get_n_cols() { return _n_cols; }  // ����һ��ͼƬ�ж�����
private:
	bool read_label_info(const std::string& filename);  // ��ȡ���ݼ���Ϣ
	bool read_images_info(const std::string& filename);
	Eigen::VectorXi _label_vec;  // ��ǩ����
	Eigen::MatrixXd _images_mat; // ͼƬ����
	std::string _label_file;     // ��ǩ�ļ���ַ
	std::string _images_file;    // ͼƬ�ļ���ַ
	int _number_of_label;        // ��ǩ����
	int _number_of_images;       // ͼƬ����
	int _n_rows;                 // һ��ͼƬ������
	int _n_cols;                 // һ��ͼƬ������
	int _max_rows;               // ��������
};

int reverse_int(int i);          // �� i ����һЩλ����
