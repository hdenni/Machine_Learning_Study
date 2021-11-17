#include <iostream>
#include <Eigen/Dense>
#include <cmath>

using namespace std;
using namespace Eigen;


static double utanh(double x) {
	return tanh(x);
}
static double upow(double x) {
	return x * x;
}
static double usigmoid(double x) {
	return 1.0 / (1 + exp(-x));
}

int* sarray(int N, int start) {
	int* ary = new int[N];
	for (int i = 0; i < N; i++) {
		ary[i] = start + i;
	}

	return ary;
}

// CNN에서 가져옴, 2D to 1D
VectorXd slice_flatten(MatrixXd X) {
	VectorXd output(X.cols() * X.rows());
	int cnt = 0;
	for (int j = 0; j < X.cols(); j++) {
		for (int i = 0; i < X.rows(); i++) {
			output(cnt++) = X(j, i);
		}
	}

	return output;
}

// W[0]: Wx, W[1]: Wh, W[2] : bias
MatrixXd* LSTM(MatrixXd X, MatrixXd* prev, MatrixXd* W) {
	MatrixXd h_prev = prev[0];
	MatrixXd c_prev = prev[1];

	MatrixXd A = X * W[0] + h_prev * W[1] + W[2];
	int H = static_cast<int>(W[0].cols() / 4);

	MatrixXd f = A(all, sarray(H, 0));
	MatrixXd g = A(all, sarray(H, H));
	MatrixXd i = A(all, sarray(H, 2 * H));
	MatrixXd o = A(all, sarray(H, 3 * H));

	f = f.unaryExpr(&usigmoid);
	g = g.unaryExpr(&utanh);
	i = i.unaryExpr(&usigmoid);
	o = o.unaryExpr(&usigmoid);

	MatrixXd c_next = f * c_prev + g * i;
	MatrixXd h_next = c_next.unaryExpr(&utanh);

	MatrixXd* result = new MatrixXd[2];
	result[0] = h_next; result[1] = c_next;

	return result;
}

// n: data 수, H: hidden layer neuron 수
// W[i][0]: Wx, W[i][1]: Wh, W[i][2]: bias
// Wx.shape: (H, H), 
MatrixXd SimpleLSTM(MatrixXd* X, MatrixXd** W, int history, int n, int H)
{
	MatrixXd* pred = new MatrixXd[2];
	pred[0] = MatrixXd::Zero(n, H); // h_pred
	pred[1] = MatrixXd::Zero(n, H); // c_pred


	// history 수 + Affine Layer(1)
	for (int i = 0; i < history; i++)
		pred = LSTM(X[i], pred, W[i]);

	// 보내기 전에 flatten 필요할 수 있음 (구조에 따라..?
	MatrixXd y_pred = slice_flatten(pred[0]);
	// Affine input:y_pred / output: y_pred에 받음

	return y_pred;
	
}

int main()
{

}
