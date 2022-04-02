#include"BpNet.h"
#include<vector>
#include<iterator>
#include<fstream>
#include<string>
bpNet::bpNet(int inputnum, int outputnum, double learningRate_) :InputNum(inputnum), OutputNum(outputnum),
learingRate(learningRate_),bpNeuronNet({})
{
	placeholder = Eigen::VectorXd::Random(inputnum);
}

bpNet::bpNet(const std::string& filename)
{
	load_model(filename);
}

//����һ�㣨���������ز㣩
void bpNet::addNeuronNet(int pnum, int num)
{
	bpNeuronNet.push_back(neuronNet(pnum,num,learingRate));
}

// ��һ������ѵ������
bool bpNet::trian_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Taget)
{
	int bp_size = bpNeuronNet.size();  // bpNet�洢���������������ʵ�ʵ�BP��������һ�㣬����������㣩

	// ��������������
	for (int index = 0; index < bp_size; index++) {
		bpNeuronNet[index].err = Eigen::VectorXd::Zero(bpNeuronNet[index].getLayerNum());
	}
	// ��ǰ����
	bpNeuronNet[0].forward_bp(Input);
	for (int index = 1; index < bp_size; index++) {
		bpNeuronNet[index].forward_bp(bpNeuronNet[index - 1].val);
	}
	Eigen::VectorXd Err = Taget - bpNeuronNet[bp_size - 1].val;  // ���һ���val��Ϊ���

	// ���򴫲�
	bpNeuronNet[bp_size - 1].err = Err;
	for (int index = bp_size - 1; index > 0; index--) {
		bpNeuronNet[index].backward_bp(bpNeuronNet[index - 1].err, bpNeuronNet[index - 1].val);
	}
	bpNeuronNet[0].backward_bp(placeholder, Input);

	return true;
}

// Ԥ�⺯��
bool bpNet::predict_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Output)
{
	// ��ǰ����
	bpNeuronNet[0].forward_bp(Input);
	for (int index = 1; index < bpNeuronNet.size(); index++) {
		bpNeuronNet[index].forward_bp(bpNeuronNet[index - 1].val);
	}
	Output = bpNeuronNet[bpNeuronNet.size() - 1].val;  // ���һ���val��Ϊ���
	return true;
}

neuronNet::neuronNet(int pnum, int num, double learningRate_) :preLayerNum(pnum), LayerNum(num),
learningRate(learningRate_)
{
	// �����ֵ
	w = Eigen::MatrixXd::Random(num, pnum);  // ����Ȩ��[preLayerNum��LayerNum]
	val = Eigen::VectorXd::Random(num);  // ������Ԫ��ֵ[LayerNum��1]
	bias = Eigen::VectorXd::Random(num);  // ���������ƫ��[LayerNum��1]
	err = Eigen::VectorXd::Zero(num);  // �����Ԥ�����[LayerNum��1](���BP�������У���Ҫ���򴫲����)
}

bool neuronNet::forward_bp(Eigen::VectorXd& fInput)
{
	// �жϾ��������Ƿ�������
	if (fInput.rows() != preLayerNum) return false;

	//��������Ȩ�ؾ���ƫ�ú�ǰһ����Ԫ��ֵ���㱾����Ԫ��ֵ
	//1.0 / (1.0 + exp(-x))��sigmoid�����
	val = (w * fInput + bias).unaryExpr([](double x) {return 1.0 / (1.0 + exp(-x)); });
	return true;
}

bool neuronNet::backward_bp(Eigen::VectorXd& preErr, Eigen::VectorXd& preVal)
{
	// ȷ����������������
	if (preErr.rows() != preVal.rows()) return false;

	Eigen::VectorXd middle_val, convert_sigmoid_val;  // �м�������������
	convert_sigmoid_val = val.unaryExpr([](double x) {return x * (1 - x); });  //x * (1 - x)��sigmoid������ĵ���|���򴫲������
	auto mid_err_matrix = convert_sigmoid_val.asDiagonal();  // ���ɶԽǾ�����ʵ������������ӦԪ�����
	middle_val = (mid_err_matrix * err).transpose();

	preErr += w.transpose() * err;  // ���򴫵������ݱ������������һ������
	w += middle_val * preVal.transpose() * learningRate;  // ����Ȩ��
	bias -= middle_val * learningRate;  // ����ƫ��bias

	return true;
}

bool bpNet::save_model(const std::string& filename)
{
	std::ofstream out(filename, std::ios::out);
	if (!out.is_open()) return false;
	int layer_num = bpNeuronNet.size();  // �������������
	out << layer_num << '\n'
		<< InputNum << '\n'  // ������Ԫ��
		<< OutputNum << '\n'  // �����Ԫ��
		<< learingRate << std::endl;  // ѧϰ��
	
	for (auto it = bpNeuronNet.cbegin(); it != bpNeuronNet.cend(); it++) {
		out << it->w.cols() << '\n'   // ���������������
			<< it->w.rows() << '\n'
		    << it->bias.rows() << std::endl;
		out << it->w << std::endl;  // �������
		out << it->bias << std::endl;
	}
	out.flush();
	out.close();
	return true;
}

bool bpNet::load_model(const std::string& filename)
{
	bpNeuronNet.clear();

	std::ifstream in(filename, std::ios::out);
	if (!in.is_open()) return false;
	int layer_num = 0;
	in >> layer_num>>InputNum>>OutputNum>>learingRate;
	placeholder = Eigen::VectorXd::Random(InputNum);
	for (int index = 0; index < layer_num; index++) {
		int w_row = 0, w_col = 0, bias_row = 0;
		if (w_row != bias_row) return false;
		in >> w_col >> w_row >> bias_row;
		addNeuronNet(w_col, w_row);  // ����һ��
		for (int row = 0; row < w_row; row++) {
			for (int col = 0; col < w_col; col++) {
				in >> bpNeuronNet[index].w(row, col);
			}
		}
		for (int row = 0; row < w_row; row++) {
			in >> bpNeuronNet[index].bias(row);
		}
	}
	in.close();
	return true;
}