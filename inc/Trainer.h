#ifndef TRAINER_H_
#define TRAINER_H_

#include "NeuralNetwork.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

class Trainer
{

private:
	NeuralNetwork *net;

	int epochs, examples, inputs, outputs;

	double learning_rate;

	void back_prop(std::vector<double>*);


public:

	// default constructor: initialize all varaibles to zero
	Trainer(){
		epochs = examples = inputs = outputs = 0;
		learning_rate = 0;
	}

	// setter functions
	void set_epochs(int epochs) {this->epochs = epochs;}
	void set_examples(int examples) {this->examples = examples;}
	void set_inputs(int inputs) {this->inputs = inputs;}
	void set_outputs(int outputs) {this->outputs = outputs;}
	void set_learning_rate(double rate){learning_rate = rate;}

	// getter functions
	int get_epochs(void){ return epochs;}
	int get_examples(void) {return examples;}
	int get_inputs(void) {return inputs;}
	int get_outputs(void) {return outputs;}
	double get_learning_rate(void) {return learning_rate;}
	NeuralNetwork* get_neural_net(void) {return net;}

	// back propagation learing
	void back_prop_learn(std::ifstream *training_file);

	// function that parses input stream and sets neural net parameters
	void import_neural_net(std::ifstream *infile);

	// writes neural net to output file
	void export_neural_net(std::ofstream *outfile);
};

#endif