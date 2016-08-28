#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <vector>
#include <fstream>
#include <iostream>
#include <string>

class NeuralNetwork {

private:

	int input_neuron_count, hidden_neuron_count, output_neuron_count;

	std::vector<double>  input_neurons, hidden_neurons, output_neurons;
	std::vector<std::vector<double>>  weights_in2hid, weights_hid2out;

	void load_neural_net(std::ifstream*);

	void feed_forward(std::vector<double>*);

	void save_weights(std::ofstream*);

	void update_activations(std::vector<double>&, std::vector<std::vector<double>>&, std::vector<double>&);
	friend class Trainer;
	friend class Tester;

public:

	NeuralNetwork(std::ifstream*);

	// setter functions
	void set_input_neuron_count(int count) {input_neuron_count = count;}
	void set_hidden_neuron_count(int count) {hidden_neuron_count = count;}
	void set_output_neuron_count(int count) {output_neuron_count = count;}

	// getter functions
	int get_input_neuron_count(void) {return input_neuron_count;}
	int get_hidden_neuron_count(void) {return hidden_neuron_count;}
	int get_output_neuron_count(void) {return output_neuron_count;}
	std::vector<std::vector<double>>* get_weights_in2hid(void) {return &weights_in2hid;}
	std::vector<std::vector<double>>* get_weights_hid2out(void) {return &weights_hid2out;}
	std::vector<double>* get_input_neurons(void) {return &input_neurons;}
	std::vector<double>* get_hidden_neurons(void) {return &hidden_neurons;}
	std::vector<double>* get_output_neurons(void) {return &output_neurons;}
};

#endif