#pragma once
#include"BpNet.h"

/* 这个头文件调用BpNet.h头文件封装的反向传播算法实现手写数字识别。
   这个文件中定义了一个类bpMnistData，对Mnist数据集进行管理，
   并定义了三个函数分别进行手写数字识别的训练、测试和预测（train_mnist、test_mnist、predict_mnist）
*/

//Mnist数据集管理类
class bpMnistData
{
public:
	bpMnistData() = default;
	//bpMnistDate(string train_file, string test);
	void load_train(const std::string& train_file);  // 将训练数据从csv文件读入至矩阵train_data
	void load_test(const std::string& test_file);  // 将测试数据从csv文件读入至矩阵test_data
	Eigen::VectorXd change_num_to_01(const int& key);  // 将一个0到9的数字，转成一个只含有0和1的向量
	int change_01_to_num(const Eigen::VectorXd& vec);  // 

	template<typename M>
	M load_csv(const std::string& path);  // 将csv文件中的数据存入一个矩阵并返回
	Eigen::MatrixXd train_data;  // 训练数据
	Eigen::MatrixXd test_data;  // 测试数据
};

void bpMnistData::load_train(const std::string& train_file)
{
	train_data = load_csv<Eigen::MatrixXd>(train_file);
}

void bpMnistData::load_test(const std::string& test_file)
{
	test_data = load_csv<Eigen::MatrixXd>(test_file);
}

Eigen::VectorXd bpMnistData::change_num_to_01(const int& key)
{
	Eigen::VectorXd vec = Eigen::VectorXd::Zero(10);
	vec[key] = 1;  // 向量的第key的值为1，其他值全为0
	return vec;
}

int bpMnistData::change_01_to_num(const Eigen::VectorXd& vec)
{
	int i = 0;
	vec.maxCoeff(&i);
	return i;  // 返回向量vec中最大值的下标
}

template<typename M>
M bpMnistData::load_csv(const std::string& path) {

	std::ifstream indata;
	indata.open(path);
	std::string line;
	std::vector<double> values;
	int rows = 0;
	std::cout << "---开始读取数据---" << std::endl;
	while (std::getline(indata, line)) {
		std::stringstream lineStream(line);
		std::string cell;
		while (std::getline(lineStream, cell, ',')) {
			// 将输入的数据全部转成0和1，这样能大大提高准确率
			values.push_back(std::stoi(cell) < 50 ? 0.0 : 1.0);  // 灰度值小于50则设为0，否则设为1
		}
		++rows;
	}
	std::cout << "---开始将数据读入矩阵---" << std::endl;
	return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size() / rows);
}

bool train_mnist(bpNet& myNet, bpMnistData& myData, int times = 10)  
{

	int InputNum = myNet.getInputNum(), OutputNum = myNet.getOutputNum();
	Eigen::VectorXd vInput(InputNum), vOutput(OutputNum);
	int myMatrix_row = myData.train_data.rows(), myMatrix_col = myData.train_data.cols();
	if (myMatrix_col - 1 != InputNum) return false;
	std::cout << "---开始训练---" << std::endl;
	// 迭代次数为times×myMatrix_row次
	for (int index = 0; index < times; index++)
	{
		for (int row = 0; row < myMatrix_row; row++)
		{
			// 将训练数据矩阵每行的第一个数转成01向量
			vOutput = myData.change_num_to_01(myData.train_data(row, 0));
			vInput = myData.train_data.block(row, 1, 1, InputNum).transpose();
			myNet.trian_bp(vInput, vOutput);
		}
		std::cout << index << " ";  // 输出训练进度
	}
	std::cout << "\n---训练完成---" << std::endl;
	return true;
}

int predict_mnist(Eigen::VectorXd& vec, bpNet& myNet)
{
	int i = -1;
	if (vec.rows() != myNet.getInputNum()) return i;
	Eigen::VectorXd outputVec = Eigen::VectorXd::Zero(myNet.getOutputNum());
	myNet.predict_bp(vec, outputVec);
	outputVec.maxCoeff(&i);
	return i;
}

double test_mnist(bpNet& myNet, bpMnistData& myData)
{
	double accuracy_rate = -1;
	int all_times = 0, accuracy_times = 0;

	int InputNum = myNet.getInputNum(), OutputNum = myNet.getOutputNum();
	Eigen::VectorXd vInput(InputNum), vOutput = Eigen::VectorXd::Zero(OutputNum);
	int myMatrix_row = myData.test_data.rows(), myMatrix_col = myData.test_data.cols();
	if (myMatrix_col - 1 != InputNum) return false;

	for (int index = 0; index < myMatrix_row; index++)
	{
		vInput = myData.test_data.block(index, 1, 1, InputNum).transpose();
		myNet.predict_bp(vInput, vOutput);  // 预测
		if (myData.change_01_to_num(vOutput) == myData.test_data(index, 0)) { accuracy_times++; } // 如果正确预测，则正确次数加一}
		all_times++;  // 预测次数加一
	}
	accuracy_rate = double(accuracy_times) / double(all_times);
	return accuracy_rate;
}