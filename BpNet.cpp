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

//增加一层（输出层或隐藏层）
void bpNet::addNeuronNet(int pnum, int num)
{
	bpNeuronNet.push_back(neuronNet(pnum,num,learingRate));
}

// 用一个样本训练网络
bool bpNet::trian_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Taget)
{
	int bp_size = bpNeuronNet.size();  // bpNet存储的神经网络层数（比实际的BP神经网络少一层，不包含输入层）

	// 将各层的误差置零
	for (int index = 0; index < bp_size; index++) {
		bpNeuronNet[index].err = Eigen::VectorXd::Zero(bpNeuronNet[index].getLayerNum());
	}
	// 向前传播
	bpNeuronNet[0].forward_bp(Input);
	for (int index = 1; index < bp_size; index++) {
		bpNeuronNet[index].forward_bp(bpNeuronNet[index - 1].val);
	}
	Eigen::VectorXd Err = Taget - bpNeuronNet[bp_size - 1].val;  // 最后一层的val即为输出

	// 反向传播
	bpNeuronNet[bp_size - 1].err = Err;
	for (int index = bp_size - 1; index > 0; index--) {
		bpNeuronNet[index].backward_bp(bpNeuronNet[index - 1].err, bpNeuronNet[index - 1].val);
	}
	bpNeuronNet[0].backward_bp(placeholder, Input);

	return true;
}

// 预测函数
bool bpNet::predict_bp(Eigen::VectorXd& Input, Eigen::VectorXd& Output)
{
	// 向前传播
	bpNeuronNet[0].forward_bp(Input);
	for (int index = 1; index < bpNeuronNet.size(); index++) {
		bpNeuronNet[index].forward_bp(bpNeuronNet[index - 1].val);
	}
	Output = bpNeuronNet[bpNeuronNet.size() - 1].val;  // 最后一层的val即为输出
	return true;
}

neuronNet::neuronNet(int pnum, int num, double learningRate_) :preLayerNum(pnum), LayerNum(num),
learningRate(learningRate_)
{
	// 赋随机值
	w = Eigen::MatrixXd::Random(num, pnum);  // 输入权重[preLayerNum×LayerNum]
	val = Eigen::VectorXd::Random(num);  // 本层神经元的值[LayerNum×1]
	bias = Eigen::VectorXd::Random(num);  // 本层的输入偏置[LayerNum×1]
	err = Eigen::VectorXd::Zero(num);  // 本层的预测误差[LayerNum×1](多层BP神经网络中，需要反向传播误差)
}

bool neuronNet::forward_bp(Eigen::VectorXd& fInput)
{
	// 判断矩阵运算是否有意义
	if (fInput.rows() != preLayerNum) return false;

	//根据输入权重矩阵、偏置和前一层神经元的值计算本层神经元的值
	//1.0 / (1.0 + exp(-x))：sigmoid激活函数
	val = (w * fInput + bias).unaryExpr([](double x) {return 1.0 / (1.0 + exp(-x)); });
	return true;
}

bool neuronNet::backward_bp(Eigen::VectorXd& preErr, Eigen::VectorXd& preVal)
{
	// 确保矩阵运算有意义
	if (preErr.rows() != preVal.rows()) return false;

	Eigen::VectorXd middle_val, convert_sigmoid_val;  // 中间变量，便于理解
	convert_sigmoid_val = val.unaryExpr([](double x) {return x * (1 - x); });  //x * (1 - x)：sigmoid激活函数的导数|反向传播激活函数
	auto mid_err_matrix = convert_sigmoid_val.asDiagonal();  // 生成对角矩阵，以实现两个向量对应元素相乘
	middle_val = (mid_err_matrix * err).transpose();

	preErr += w.transpose() * err;  // 反向传递误差，根据本层误差计算出上一层的误差
	w += middle_val * preVal.transpose() * learningRate;  // 更新权重
	bias -= middle_val * learningRate;  // 更新偏置bias

	return true;
}

bool bpNet::save_model(const std::string& filename)
{
	std::ofstream out(filename, std::ios::out);
	if (!out.is_open()) return false;
	int layer_num = bpNeuronNet.size();  // 读入神经网络层数
	out << layer_num << '\n'
		<< InputNum << '\n'  // 输入神经元数
		<< OutputNum << '\n'  // 输出神经元数
		<< learingRate << std::endl;  // 学习率
	
	for (auto it = bpNeuronNet.cbegin(); it != bpNeuronNet.cend(); it++) {
		out << it->w.cols() << '\n'   // 读入参数的行列数
			<< it->w.rows() << '\n'
		    << it->bias.rows() << std::endl;
		out << it->w << std::endl;  // 读入参数
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
		addNeuronNet(w_col, w_row);  // 插入一层
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