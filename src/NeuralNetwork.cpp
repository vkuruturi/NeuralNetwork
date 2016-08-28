#include "NeuralNetwork.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

// sigmoid definitions
#define SIGMOID(X) 1/(1+exp(-X))
#define SIGDERIV(X) SIGMOID(X)*1-SIGMOID(X)


NeuralNetwork::NeuralNetwork(std::ifstream *in){
	load_neural_net(in);
}

void NeuralNetwork::load_neural_net(std::ifstream *in){
	*in >> input_neuron_count >> hidden_neuron_count >> output_neuron_count;
	
	// init 2d array for input->hidden layer weights
	std::vector<double> v(hidden_neuron_count+1,0);
	weights_in2hid.resize(input_neuron_count + 1, v);

	// init 2d array for hidden->output layer weights
	v.resize(output_neuron_count+1,0);
	weights_hid2out.resize(hidden_neuron_count+1,v);
	
	// load input->hidden layer weights
	for (int j = 0; j <= hidden_neuron_count; ++j)
	{
		for (int i = 0; i <= input_neuron_count; ++i)
		{
			*in >> weights_in2hid[i][j];
		}
	}

	
	// load hidden->output layer weights
	for (int j = 0; j <= output_neuron_count; ++j)
	{
		for (int i = 0; i <= hidden_neuron_count; ++i)
		{
			*in >> weights_hid2out[i][j];
		}
	}

	input_neurons.resize(input_neuron_count+1,0);
	hidden_neurons.resize(hidden_neuron_count+1,0);
	output_neurons.resize(output_neuron_count+1,0);

	input_neurons[0] = -1;
	hidden_neurons[0] = -1;
}

void NeuralNetwork::feed_forward(const std::vector<double> &input){
	// error check: match input size to size of input neurons
	if (input_neurons.size() != input.size() +1)
	{
		std::cerr << "NeuralNetwork::feed_forward: input size does not match! Input neuron count (w/o bias): "  << input_neuron_count << " input vector size: " <<input.size() << ". Exiting..." << std::endl;
		exit(1);
	}
/*
	// fill all neuron values with 0
	std::fill(input_neurons.begin(),input_neurons.end(),0);
	std::fill(hidden_neurons.begin(), hidden_neurons.end(),0);
	std::fill(output_neurons.begin(), output_neurons.end(),0);

	// set bias neurons to -1
	input_neurons[0] = -1;
	hidden_neurons[0] = -1;

	// fill in input neurons
	for (int i = 1; i <input_neurons.size(); ++i){
		input_neurons[i] = input[i];
	}

	update_activations(input_neurons,weights_in2hid,hidden_neurons);
	update_activations(hidden_neurons,weights_hid2out,output_neurons);*/
	return;
}

void NeuralNetwork::update_activations(std::vector<double> &previous, std::vector<std::vector<double>> &weights, std::vector<double> &next){
	double d;
	// do not update bias neuron of next layer, so start from j=1
	for (int j = 1; j < next.size(); ++j)
	{
		d = 0;
		for (int i = 0; i < previous.size(); ++i)
		{
			d += previous[i] *weights[i][j];
		}
		next[j] = SIGMOID(d);
	}
}

void NeuralNetwork::save_weights(std::ofstream *out){
	*out << input_neuron_count << " " << hidden_neuron_count << " " << output_neuron_count << std::endl;

	for (int j = 1; j <= hidden_neuron_count; ++j)
	{
		*out << std::setprecision(3) << std::fixed << weights_in2hid[0][j];
		for (int i = 0; i < input_neuron_count; ++i)
		{
			*out << std::setprecision(3) << std::fixed << weights_in2hid[i][j];
		}
		*out << std::endl;
	}

	for (int j = 1; j <= output_neuron_count; ++j)
	{
		*out << std::setprecision(3) << std::fixed << weights_hid2out[0][j];
		for (int i = 0; i < hidden_neuron_count; ++i)
		{
			*out << std::setprecision(3) << std::fixed << weights_hid2out[i][j];
		}
		*out << std::endl;
	}

}
