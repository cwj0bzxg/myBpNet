#include"BpData.h"

int reverse_int(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;  // 位运算
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return static_cast<int>(ch1 << 24) + static_cast<int>(ch2 << 16)
		+ static_cast<int>(ch3 << 8) + static_cast<int>(ch4);
}

bpData::bpData(const std::string& label_filename, const std::string& images_filename,int max_rows):_label_file(label_filename),
_images_file(images_filename),_number_of_label(0),_number_of_images(0),_n_rows(0),_n_cols(0),_max_rows(max_rows)
{
	read_label_info(label_filename);  // 读取数据集信息
	read_images_info(images_filename);
	_label_vec.resize(this->get_number_of_label());  // 调整矩阵大小
	_images_mat.resize(this->get_number_of_images(), _n_rows * _n_cols);
}

bool bpData::read_label_info(const std::string& filename)
{
	std::ifstream in(filename, std::ios::binary);
	if (!in.is_open()) return false;
	int magic_number = 0;
	int number_of_label = 0;
	in.read((char*)&magic_number, sizeof(magic_number));
	in.read((char*)&number_of_label, sizeof(number_of_label));
	//magic_number = reverse_int(magic_number);
	_number_of_label = reverse_int(number_of_label);

	in.close();

	return true;
}

bool bpData::read_images_info(const std::string& filename)
{
	std::ifstream in(filename, std::ios::binary);
	if (!in.is_open()) return false;
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;
	unsigned char label;
	in.read((char*)&magic_number, sizeof(magic_number));
	in.read((char*)&number_of_images, sizeof(number_of_images));
	in.read((char*)&n_rows, sizeof(n_rows));
	in.read((char*)&n_cols, sizeof(n_cols));
	//magic_number = reverse_int(magic_number);
	_number_of_images = reverse_int(number_of_images);
	_n_rows = reverse_int(n_rows);
	_n_cols = reverse_int(n_cols);

	in.close();
}

bool bpData::read_label()
{
	std::ifstream in(_label_file, std::ios::binary);
	//if (!in.is_open()) return Eigen::VectorXi();
	int magic_number = 0;
	int number_of_label = 0;
	in.read((char*)&magic_number, sizeof(magic_number));
	in.read((char*)&number_of_label, sizeof(number_of_label));
	magic_number = reverse_int(magic_number);
	number_of_label = reverse_int(number_of_label);

	number_of_label = std::min(number_of_label, _max_rows);
	//Eigen::VectorXi label_vector(number_of_label);

	for (int index = 0; index < number_of_label; index++) {
		unsigned char val = 0;
		in.read((char*)&val, sizeof(val));
		//label_vector[index] = static_cast<int>(val);
		_label_vec[index] = static_cast<int>(val);
	}

	in.close();
	return true;
}

bool bpData::read_images()
{
	std::ifstream in(_images_file, std::ios::binary);
	//if (!in.is_open()) return false;
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;
	unsigned char label;
	in.read((char*)&magic_number, sizeof(magic_number));
	in.read((char*)&number_of_images, sizeof(number_of_images));
	in.read((char*)&n_rows, sizeof(n_rows));
	in.read((char*)&n_cols, sizeof(n_cols));
	//magic_number = reverse_int(magic_number);
	number_of_images = reverse_int(number_of_images);
	n_rows = reverse_int(n_rows);
	n_cols = reverse_int(n_cols);

	number_of_images = std::min(number_of_images, _max_rows);
	//Eigen::MatrixXd images_matrix(number_of_images, n_rows * n_cols);

	int num = number_of_images * n_rows * n_cols;
	int cols = n_rows * n_cols;
	for (int index = 0; index < num; index++) {
		unsigned char val = 0;
		in.read((char*)&val, sizeof(val));
		_images_mat(index / cols, index % cols) = (static_cast<int>(val) < 50 ? 0.0 : 1.0);
	}

	in.close();

	return true;
}

bool train_bp(bpNet& myNet, bpData& myData, int epoch)
{
	if (myNet.getInputNum() != myData.get_n_rows() * myData.get_n_cols()) return false;

	Eigen::VectorXd input_vec(myNet.getInputNum());
	Eigen::VectorXd taget_vec = Eigen::VectorXd::Zero(myNet.getOutputNum());

	if (!myData.read_label() || !myData.read_images()) return false;

	int rows = myData.get_number_of_label();
	int cols = myData.get_n_rows() * myData.get_n_cols();

	for (int index = 0; index < epoch; index++) {
		std::cout << "第" << index + 1 << "次训练" << std::endl;
		for (int row = 0; row < rows; row++) {
			input_vec = myData._images_mat.block(row, 0, 1, cols).transpose();
			taget_vec[myData._label_vec[row]] = 1.0;
			myNet.trian_bp(input_vec, taget_vec);
			taget_vec[myData._label_vec[row]] = 0.0;
		}
	}
	return true;
}

double test_bp(bpNet& myNet, bpData& myData)
{
	Eigen::VectorXd input_vec(myNet.getInputNum());
	Eigen::VectorXd output_vec = Eigen::VectorXd::Zero(myNet.getOutputNum());

	if (!myData.read_label() || !myData.read_images()) return false;

	int rows = myData.get_number_of_label();
	int cols = myData.get_n_rows() * myData.get_n_cols();

	double accuracy_times = 0, all_times = 0;  // 记录

	for (int row = 0; row < rows; row++) {
		input_vec = myData._images_mat.block(row, 0, 1, cols).transpose();
		myNet.predict_bp(input_vec, output_vec);
		int i = 0;
		output_vec.maxCoeff(&i);
		if (myData._label_vec[row] == i) accuracy_times++;
		all_times++;
	}
	return accuracy_times / all_times;
}

int predict_bp(bpNet& myNet, Eigen::VectorXd& Input)
{
	int i = 0;
	Eigen::VectorXd output(myNet.getOutputNum());
	myNet.predict_bp(Input, output);
	output.maxCoeff(&i);
	return i;
}