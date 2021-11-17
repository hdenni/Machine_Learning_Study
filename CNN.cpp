#include <iostream>
#include <Eigen/Dense>
#include <random>


using namespace std;
using namespace Eigen;

MatrixXd* convolution(MatrixXd* input, MatrixXd** weight, int FN, int C, int stride, int padding);
MatrixXd* pooling(MatrixXd* X, int n, int PH, int PW, int stride, int padding);
MatrixXd im2col(MatrixXd* X, int C, int OH, int OW, int FH, int FW);
VectorXd slice_flatten(MatrixXd X, int y, int x, int FH, int FW);

int main() {
	default_random_engine generator;
	normal_distribution<double> distribution(0.0, 1.0);

	int C = 3;
	int W = 5, H = 5;

	// set Input
	// data 1개, channel 3개, 5x5
	MatrixXd* input = new MatrixXd[C];
	for (int c = 0; c < C; c++) {
		MatrixXd temp(H, W);
		for (int y = 0; y < H; y++) {
			for (int x = 0; x < W; x++) {
				temp(y, x) = c * H * W + y * 5 + x + 1;
			}
		}
		input[c] = temp;
	}

	cout << "INPUT" << endl;
	cout << input[0] << endl << endl;
	cout << input[1] << endl << endl;
	cout << input[2] << endl << endl;


	// set weight
	int FN = 2; // Filter number
	int FW = 2, FH = 2;

	// memory allocation
	MatrixXd** weight = new MatrixXd * [FN];
	for (int i = 0; i < FN; i++) {
		weight[i] = new MatrixXd[C];
		for (int j = 0; j < C; j++) {
			MatrixXd temp2(2, 2);
			weight[i][j] = temp2;
		}
	}

	// weight 할당
	/*
	weight[0][0] << 1, 2, 0, 1;
	weight[0][1] << 2, 0, -1, 1;
	weight[0][2] << 0, 2, 1, -1;

	weight[1][0] << 0, 1, 0, -1;
	weight[1][1] << 1, 0.5, -0.2, 0.3;
	weight[1][2] << 0, 2, -3, 1;
	 */

	 // 정규분포를 활용한 초기화
	for (int i = 0; i < FN; i++) {
		for (int c = 0; c < C; c++) {
			for (int y = 0; y < FH; y++) {
				for (int x = 0; x < FW; x++) {
					weight[i][c](y, x) = distribution(generator);
				}
			}
		}
	}

	MatrixXd* test = convolution(input, weight, FN, C, 1, 0);
	cout << "Convolution Result" << endl;
	for (int i = 0; i < FN; i++) {
		cout << test[i] << endl << endl;
	}

	cout << "Pooling Result" << endl;
	MatrixXd* test2 = pooling(test, FN, 2, 2, 2, 0);
	for (int i = 0; i < FN; i++) {
		cout << test2[i] << endl << endl;
	}

	delete test;
	delete test2;

	return 0;
}

MatrixXd* convolution(MatrixXd* input, MatrixXd** weight, int FN, int C, int stride, int padding) {
	int H = input[0].rows();
	int W = input[0].cols();

	int FH = weight[0][0].rows();
	int FW = weight[0][0].cols();

	int OH = int((H + 2 * padding - FH) / stride) + 1;
	int OW = int((W + 2 * padding - FW) / stride) + 1;

	MatrixXd col = im2col(input, C, OH, OW, FH, FW);
	MatrixXd* col_w = new MatrixXd[FN];

	// 필터 전개
	// 4D(FN, C, FH, FW) to 2D(FN, C*FH*FW)
	for (int w = 0; w < FN; w++) {
	
		VectorXd temp(FH * FW * C);

		// 3D(C, FH, FW) to 1D(C*FH*FW)
		for (int c = 0; c < C; c++) {
			// 1D weight(FH*FW) vector를 가로로 concat -> output: Vector(1D)
			VectorXd v = slice_flatten(weight[w][c], 0, 0, FH, FW); // Matrix(FH, FW) -> Vector(FH*FW)
			for (int i = 0; i < FH * FW; i++)
				temp(c * FH * FW + i) = v(i);
		}
		// 하나의 Filter(FN)마다 한개의 row
		col_w[w] = temp.matrix();
	} // -> output: 2D Matrix


	// Convolution 연산 + reshape
	MatrixXd* output = new MatrixXd[FN];
	for (int i = 0; i < FN; i++) {
		MatrixXd temp(OH, OW);
		output[i] = temp;

		MatrixXd result = col * col_w[i];

		for (int y = 0; y < OH; y++) {
			for (int x = 0; x < OW; x++) {
				output[i](y, x) = result(y * OH + x, 0);
			}
		}
	}
	return output;
}

// max pooling
MatrixXd* pooling(MatrixXd* X, int n, int PH, int PW, int stride, int padding) {
	MatrixXd* output = new MatrixXd[n];

	int IH = X[0].rows();
	int IW = X[0].cols();

	int OH = (int)(1 + (IH - PH) / stride);
	int OW = (int)(1 + (IW - PW) / stride);

	for (int i = 0; i < n; i++) {
		MatrixXd temp(OH, OW);
		for (int y = 0; y < IH; y += stride) {
			for (int x = 0; x < IW; x += stride) {
				VectorXd m = slice_flatten(X[i], y, x, PH, PW);
				//cout << m.matrix().transpose() << endl;
				double max = 0;
				for (int t = 0; t < m.size(); t++) {
					if (max < m(t)) max = m(t);
				}
				temp((int)(y / PH), (int)(x / PW)) = max;
			}
		}
		output[i] = temp;
	}

	return output;
}

// 3D to 2D
MatrixXd im2col(MatrixXd* X, int C, int OH, int OW, int FH, int FW) {
	MatrixXd output(OH * OW, FH * FW * C);
	int cnt = 0;

	for (int y = 0; y < OH; y++) {
		for (int x = 0; x < OW; x++) {
			VectorXd* vec = new VectorXd[C];
			for (int c = 0; c < C; c++)
				vec[c] = slice_flatten(X[c], y, x, FH, FW);

			int r = vec[0].rows();
			VectorXd v(r * C);
			for (int i = 0; i < r * C; i++) {
				v(i) = vec[int(i / r)](i% r);
			}

			output.row(cnt++) = v;
		}
	}
	return output;
}

VectorXd slice_flatten(MatrixXd X, int y, int x, int FH, int FW) {
	VectorXd output(FH * FW);
	int cnt = 0;
	for (int j = y; j < y + FH; j++) {
		for (int i = x; i < x + FW; i++) {
			output(cnt++) = X(j, i);
		}
	}

	return output;
}
