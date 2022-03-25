#pragma once
#include"BpNet.h"

/* ���ͷ�ļ�����BpNet.hͷ�ļ���װ�ķ��򴫲��㷨ʵ����д����ʶ��
   ����ļ��ж�����һ����bpMnistData����Mnist���ݼ����й���
   �����������������ֱ������д����ʶ���ѵ�������Ժ�Ԥ�⣨train_mnist��test_mnist��predict_mnist��
*/

//Mnist���ݼ�������
class bpMnistData
{
public:
	bpMnistData() = default;
	//bpMnistDate(string train_file, string test);
	void load_train(const std::string& train_file);  // ��ѵ�����ݴ�csv�ļ�����������train_data
	void load_test(const std::string& test_file);  // ���������ݴ�csv�ļ�����������test_data
	Eigen::VectorXd change_num_to_01(const int& key);  // ��һ��0��9�����֣�ת��һ��ֻ����0��1������
	int change_01_to_num(const Eigen::VectorXd& vec);  // 

	template<typename M>
	M load_csv(const std::string& path);  // ��csv�ļ��е����ݴ���һ�����󲢷���
	Eigen::MatrixXd train_data;  // ѵ������
	Eigen::MatrixXd test_data;  // ��������
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
	vec[key] = 1;  // �����ĵ�key��ֵΪ1������ֵȫΪ0
	return vec;
}

int bpMnistData::change_01_to_num(const Eigen::VectorXd& vec)
{
	int i = 0;
	vec.maxCoeff(&i);
	return i;  // ��������vec�����ֵ���±�
}

template<typename M>
M bpMnistData::load_csv(const std::string& path) {

	std::ifstream indata;
	indata.open(path);
	std::string line;
	std::vector<double> values;
	int rows = 0;
	std::cout << "---��ʼ��ȡ����---" << std::endl;
	while (std::getline(indata, line)) {
		std::stringstream lineStream(line);
		std::string cell;
		while (std::getline(lineStream, cell, ',')) {
			// �����������ȫ��ת��0��1�������ܴ�����׼ȷ��
			values.push_back(std::stoi(cell) < 50 ? 0.0 : 1.0);  // �Ҷ�ֵС��50����Ϊ0��������Ϊ1
		}
		++rows;
	}
	std::cout << "---��ʼ�����ݶ������---" << std::endl;
	return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size() / rows);
}

bool train_mnist(bpNet& myNet, bpMnistData& myData, int times = 10)  
{

	int InputNum = myNet.getInputNum(), OutputNum = myNet.getOutputNum();
	Eigen::VectorXd vInput(InputNum), vOutput(OutputNum);
	int myMatrix_row = myData.train_data.rows(), myMatrix_col = myData.train_data.cols();
	if (myMatrix_col - 1 != InputNum) return false;
	std::cout << "---��ʼѵ��---" << std::endl;
	// ��������Ϊtimes��myMatrix_row��
	for (int index = 0; index < times; index++)
	{
		for (int row = 0; row < myMatrix_row; row++)
		{
			// ��ѵ�����ݾ���ÿ�еĵ�һ����ת��01����
			vOutput = myData.change_num_to_01(myData.train_data(row, 0));
			vInput = myData.train_data.block(row, 1, 1, InputNum).transpose();
			myNet.trian_bp(vInput, vOutput);
		}
		std::cout << index << " ";  // ���ѵ������
	}
	std::cout << "\n---ѵ�����---" << std::endl;
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
		myNet.predict_bp(vInput, vOutput);  // Ԥ��
		if (myData.change_01_to_num(vOutput) == myData.test_data(index, 0)) { accuracy_times++; } // �����ȷԤ�⣬����ȷ������һ}
		all_times++;  // Ԥ�������һ
	}
	accuracy_rate = double(accuracy_times) / double(all_times);
	return accuracy_rate;
}