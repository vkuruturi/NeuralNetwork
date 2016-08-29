#ifndef TESTER_H_
#define TESTER_H_

#include "NeuralNetwork.h"
#include <vector>
#include <iostream>

class Tester{

private:
	NeuralNetwork *net;

	std::vector<double> input;
	std::vector<std::vector<bool>> expected_output;
	std::vector<std::vector<bool>> classified;

public:

	Tester(std::ifstream *in);

	void test_network(std::ifstream*);
	void output_metrics(std::ofstream*);
};

#endif