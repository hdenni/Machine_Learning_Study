#include <iostream>
#include <random>

#define MAX_NEURON 10
#define MAX_LAYER 10

#define INPUT_LAYER 0
#define HIDDEN_LAYER 1
#define OUTPUT_LAYER 2

// for index
#define INPUT 0
#define OUTPUT 3
   
/*
neural network code without library
*/

using namespace std;

class Neuron {
private:
	double z, a; //z: before activation, a:after activation
	double* weight; // 다음 뉴런의 결과값을 구할 때 곱해줄 weight
	int weight_num; // weight 수
	double bias;

	double grad_w; // weight gradient값, bias gradient값을 구할 때 사용하는 변수
public:
	Neuron() {
		z = 0; a = 0;
		weight_num = 1;
		weight = new double[weight_num];
		grad_w = 0;
		bias = 0;
	}
	void set_z(double x) { z = x; }
	void set_a(double x) { a = x; }
	void set_weight(double* w) { weight = w; }
	void set_weight_num(int n) { weight_num = n; }
	void set_grad_w(double gw) { grad_w = gw; }
	void set_bias(double b) { bias = b; }
	double get_z() { return z; }
	double get_a() { return a; }
	double* get_weight() { return weight; }
	double get_weight_num() { return weight_num; }
	double get_grad_w() { return grad_w; }
	double get_bias() { return bias; }
};

class Layer: Neuron {
private:
	Neuron neuron[MAX_NEURON];
	int neuron_count;
	int layer_type;
	// 0: input layer, 1: hidden layer, 2: output layer

	//double* bias;
	int act_type; //activation function type
	// 0: identify, 1: relu

public:
	Layer() {
		neuron_count = 0;
		layer_type = HIDDEN_LAYER; 
		act_type = 0; 
	}
	void set_layer_type(int t) { layer_type = t; }
	void set_act_type(int a) { act_type = a; }
	
	void set_neuron(Neuron n) { neuron[neuron_count++] = n; }
	void set_neuron_i(Neuron n, int i) { neuron[i] = n; }
	void set_neuron_value(int i, double v) {
		neuron[i].set_z(v);
		if (act_type == OUTPUT_LAYER) neuron[i].set_a(identify(v));
		else neuron[i].set_a(relu(v));
	}
	//void set_bias(double* b) { bias = b; }

	int get_layer_type() { return layer_type; }
	int get_act_type() { return act_type; }
	int get_neuron_count() { return neuron_count; }
	Neuron* get_neuron() { return neuron; }
	Neuron get_neuron_i(int i) { return neuron[i]; }
	//double* get_bias() { return bias; }

	/* activation */
	double relu(double x) { return x > 0 ? x : 0; }
	double identify(double x) { return x; }
};





// Global variable
Layer layer[4]; //input layer, hidden layer 1, hidden layer 2, output layer
int lnum = 4; // layer number

double input[4] = { 1.0, -3.0, 2.5, -4.2 };
double bias[4] = { 0.1, 0.1, 0.5, 0.3 }; 




// user define function
Layer init_layer(int type, int neuron_num, int weight_num, double b);
double* alloc_weight(int n);
double feedforward();
void backprop(double answer, double learning_rate);
double calculate_value(Layer pLayer, double b, int n);
double inv_relu(double x);
double inv_identify(double x);
void print_wb();
void print_za();




int main() {
	// set layer
	layer[INPUT]  = init_layer(INPUT_LAYER,  4, 3, bias[0]);
	layer[1]	  = init_layer(HIDDEN_LAYER, 3, 2, bias[1]);
	layer[2]	  = init_layer(HIDDEN_LAYER, 2, 1, bias[2]);
	layer[OUTPUT] = init_layer(OUTPUT_LAYER, 1, 1, bias[3]);
	
	double y;
	double answer = 0.2;
	double learning_rate = 0.1;

	int iter = 10;
	for (int i = 0; i < iter; i++) {
		// feedforward
		y = feedforward();
		printf("%d: %lf\n",i+1, y);
		
		// backpropagate
		backprop(answer, learning_rate);
	}

	cout << "Result: " << feedforward() << endl;
	
	return 0;
}






Layer init_layer(int type, int neuron_num, int weight_num, double b) {
	Layer l;
	for (int i = 0; i < neuron_num; i++) {
		Neuron n;
		if (type != OUTPUT_LAYER) {
			n.set_weight(alloc_weight(weight_num));
			n.set_weight_num(weight_num);
		}
		if (type == INPUT_LAYER) n.set_a(input[i]);
		if (type != INPUT_LAYER) n.set_bias(b);
		l.set_neuron(n);
	}
	l.set_layer_type(type);

	return l;
}

double* alloc_weight(int n) {
	default_random_engine generator;
	normal_distribution<double> distribution(0.0, 1.0);
	double* w = new double[n];
	for (int i = 0; i < n; i++) {
		w[i] = distribution(generator);
	}
	return w;
}

double feedforward() {
	for (int h = 0; h < OUTPUT; h++) {
		Layer pLayer = layer[h];
		Layer pNextLayer = layer[h + 1];

		int pNextnum = pNextLayer.get_neuron_count();

		for (int n = 0; n < pNextnum; n++) {
			double b = pNextLayer.get_neuron_i(n).get_bias();
			layer[h + 1].set_neuron_value(n, calculate_value(pLayer, b, n));
		}
	}
	return layer[OUTPUT].get_neuron()[0].get_a();
}

void backprop(double answer, double learning_rate) {
	for (int l = lnum - 1; l >= 0; l--) { // 레이어
		
		// layer 4
		if (l == lnum - 1) {
			Neuron output = layer[OUTPUT].get_neuron_i(0);
			output.set_grad_w(output.get_a() - answer);
			layer[OUTPUT].set_neuron_i(output, 0);

			continue;
		}

		// else
		Neuron* now = layer[l].get_neuron();
		for (int n = 0; n < layer[l].get_neuron_count(); n++) { // 뉴런
			Neuron* prev = layer[l + 1].get_neuron();
			double grad = 0;
			for (int m = 0; m < layer[l + 1].get_neuron_count(); m++) {
				grad += prev[m].get_grad_w() * now[n].get_weight()[m] * inv_relu(prev[m].get_z());
			}
			now[n].set_grad_w(grad);

			// Update Weight & Bias
			double* new_weight = now[n].get_weight();
			double new_bias = now[n].get_bias() - now[n].get_grad_w() * learning_rate;
			for (int w = 0; w < now[n].get_weight_num(); w++) {
				new_weight[w] -= now[n].get_a() * prev[w].get_grad_w() * learning_rate;
			}
			now[n].set_weight(new_weight);
			now[n].set_bias(new_bias);

			layer[l].set_neuron_i(now[n], n);
		}
	}

	return;
}

double calculate_value(Layer pLayer, double b, int n) {
	Neuron* neuron = pLayer.get_neuron();
	int pNum = pLayer.get_neuron_count();
	double sum = 0;

	for (int i = 0; i < pNum; i++) {
		sum += neuron[i].get_a() * neuron[i].get_weight()[n];
	}
	sum += b;

	return sum;
}

// inverse function
double inv_relu(double x) {
	return x > 0 ? 1 : 0;
}
double inv_identify(double x) {
	return 1;
}


/* 확인용 Function (print) */
// print weight & bias
void print_wb() {
	for (int l = 0; l < 4; l++) {
		// print weight
		cout << "weight" << endl;

		Neuron* neuron = layer[l].get_neuron();
		for (int n = 0; n < layer[l].get_neuron_count(); n++) {
			double* weight = neuron[n].get_weight();
			for (int w = 0; w < neuron[n].get_weight_num(); w++) {
				cout << weight[w] << " ";
			}
			cout << endl;
		}

		cout << "bias" << endl;
		for (int n = 0; n < layer[l].get_neuron_count(); n++) {
			cout << neuron[n].get_bias() << " ";
		}
		cout << endl;
		cout << endl;
	}
}

// print z(before activation) & a(after activation)
void print_za() {
	int temp[4] = { 4, 3, 2, 1 };

	for (int l = 0; l < 4; l++) {
		cout << "layer " << l << endl;
		Neuron* neuron = layer[l].get_neuron();
		for (int n = 0; n < layer[l].get_neuron_count(); n++) {
			cout <<  neuron[n].get_z() << " " << neuron[n].get_a() << endl;
		}
	}
}
