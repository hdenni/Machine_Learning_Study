#include <iostream>
#include <random>
#include <Eigen/Dense>

#define INPUT_LAYER 0
#define HIDDEN_LAYER 1
#define OUTPUT_LAYER 2

using namespace std;
using namespace Eigen;

/* global variables */
int layer[4] = { 4, 3, 2, 1 };

/* weight */
MatrixXd w1, w2, w3;
/* bias */
MatrixXd b1(1, layer[1]);
MatrixXd b2(1, layer[2]);
MatrixXd b3(1, layer[3]);
/* result(before activation) */
MatrixXd z1, z2, z3;
/* result(after activation) */
MatrixXd a1, a2; // 마지막 계산값은 y

/* defin function */
double relu(double);
double inv_relu(double);
double identify(double);
double inv_identify(double);
MatrixXd predict(MatrixXd);
MatrixXd set_weight(int, int);
double loss(MatrixXd, MatrixXd);
MatrixXd gradient(MatrixXd);
void print_shape(MatrixXd);


int main()
{
	// input
	MatrixXd input(1, layer[0]);
	input << 1.0, -3.0, 2.5, -4.2;

	// output
	MatrixXd answer(1, 1);
	answer << 0.2;

	double lnum = 4;
	double learning_rate = 0.1;	
	
	/* init weight & bias */
	w1 = set_weight(layer[0], layer[1]);
	w2 = set_weight(layer[1], layer[2]);
	w3 = set_weight(layer[2], layer[3]);

	cout << w1 << endl;
	cout << endl;
	cout << w2 << endl;
	cout << endl;
	cout << w3 << endl;
	cout << endl;

	for (int i = 0; i < layer[1]; i++)
		b1(0, i) = 0.1;
	for (int i = 0; i < layer[2]; i++)
		b2(0, i) = 0.5;
	for (int i = 0; i < layer[3]; i++)
		b3(0, i) = 0.3;

	/*MatrixXd y = predict(input);
	cout << y(0, 0);*/

	MatrixXd y;
	MatrixXd dout;

	int iter = 10;
	
	for (int i = 0; i < iter; i++) {
		/* feedforward */
		y = predict(input);
		cout << y(0, 0) << ", ";

		/* backpropagation */
      	// dout: 각 뉴런의 gradient
        // grad_w: dW
		MatrixXd dout = y - answer; // 1x1
		MatrixXd grad_w3 = a2.transpose() * dout;
		// 2x1 * 1x1 = 2x1(w3)

		// dout = 현재 Weight * 이전(after) 레이어에서 온 gradient 값(dout) * 다음(before)레이어의 결과값의 inverse activation 값 
		MatrixXd dout2 = ((w3 * dout.transpose()).transpose().array() * z2.unaryExpr(&inv_relu).array()).matrix();
		// (2x1 * 1x1).T = 1x2
		// 1x2 
		// 둘을 각각 곱하여 1x2의 matrix 출력
		
		// gradient weight = weight 기준으로 왼쪽에 있는 activation result * 이전 레이어에서 온 gradient(오른쪽)
		MatrixXd grad_w2 = a1.transpose() * dout2;
		// 3x1 * 1x2 = 3x2(w2)

		MatrixXd dout3 = ((w2 * dout2.transpose()).transpose().array() * z1.unaryExpr(&inv_relu).array()).matrix();
		// (3x2 * 2x1).T = 1x3
		// 1x3
		// 둘을 각각 곱하여 1x3의 matirx 출력

		MatrixXd grad_w1 = input.transpose() * dout3;
		// 4x1 * 1x3 = 4x3 (w1)
		w1 -= grad_w1 * learning_rate;
		w2 -= grad_w2 * learning_rate;
		w3 -= grad_w3 * learning_rate;

		b1 -= dout3 * learning_rate; // 3*1
		b2 -= dout2 * learning_rate; // 2*1
		b3 -= dout * learning_rate; // 1*1
		
	}
	y = predict(input);
	cout << y(0, 0) << endl;

}
/* activation function */
double relu(double x) {
	return x > 0 ? x : 0;
}
double inv_relu(double x) {
	return x > 0 ? 1 : 0;
}
double identify(double x) {
	return x;
}
double inv_identify(double x) {
	return 1;
}
// feedforward
MatrixXd predict(MatrixXd input) {
	z1 = input * w1 + b1;
	a1 = z1.unaryExpr(&relu);

	z2 = a1 * w2 + b2;
	a2 = z2.unaryExpr(&relu);

	z3 = a2 * w3 + b3;
	MatrixXd y = z3.unaryExpr(&identify);

	return y;
}
// initialize weight
MatrixXd set_weight(int rows, int cols) {
	default_random_engine generator;
	normal_distribution<double> distribution(0.0, 1.0);
	MatrixXd w(rows, cols);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			w(i, j) = distribution(generator);
		}
	}
	return w;
}
// calculate loss
double loss(MatrixXd input, MatrixXd answer) {
	MatrixXd pred = predict(input);
	MatrixXd sum(answer.rows(), answer.cols());

	sum = 0.5 * ((answer - pred).array() * (answer - pred).array());
	return sum(0, 0);
}
void print_shape(MatrixXd x) {
	cout << x.rows() << "X" << x.cols() << endl;
}
