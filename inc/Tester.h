#ifndef TESTER_H_
#define TESTER_H_

#include "NeuralNetwork.h"
#include <vector>
#include <iostream>

class Tester{

private:
	NeuralNetwork *net;

	std::vector<std::vector<double>> metrics;
	double macro_metrics[4], micro_metrics[4];

	void compare_outputs(std::vector<bool>*);
	void calculate_metrics(void);


public:

	Tester(std::ifstream *in);

	void test_network(std::ifstream*);
	void output_metrics(std::ofstream*);
};

#endif