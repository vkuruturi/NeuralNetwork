#include "NeuralNetwork.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <sstream>

// sigmoid definitions
#define SIGMOID(X) 1/(1+exp(-X))
#define SIGDERIV(X) (SIGMOID(X) * (1-SIGMOID(X)))

NeuralNetwork::NeuralNetwork(std::ifstream *in){
	load_neural_net(in);
}

// load neural net from filestream
void NeuralNetwork::load_neural_net(std::ifstream *in){
	*in >> input_neuron_count >> hidden_neuron_count >> output_neuron_count;
	
	// init 2d array for input->hidden layer weights
	std::vector<double> v((unsigned int) input_neuron_count+1,0);
	weights_in2hid.resize((unsigned int) hidden_neuron_count, v);

	// init 2d array for hidden->output layer weights
	v.resize((unsigned int) hidden_neuron_count+1,0);
	weights_hid2out.resize((unsigned int) output_neuron_count,v);
    std::string line;
    double d = 0;
    std::getline(*in, line);

	// load input->hidden layer weights
	for (int i = 0; i < hidden_neuron_count; ++i)
	{
        std::stringstream ss;
        std::getline(*in,line);
        //std::cout << line << std::endl;
        ss.str(line);
		for (int j = 0; j <= input_neuron_count; ++j)
		{
			ss >> d;
            weights_in2hid[i][j] = d;
		}
	}

	
	// load hidden->output layer weights
	for (int i = 0; i < output_neuron_count; ++i)
	{
        std::stringstream ss;
        std::getline(*in,line);
        ss.str(line);
        ss.clear();
		for (int j = 0; j <= hidden_neuron_count; ++j)
		{
			ss >> weights_hid2out[i][j];
		}
	}

	// resize neuron and activation vectors
	input_neurons.resize((unsigned int) input_neuron_count,0);
	hidden_neurons.resize((unsigned int) hidden_neuron_count,0);
	output_neurons.resize((unsigned int) output_neuron_count,0);

    input_activations.resize((unsigned int) input_neuron_count,0);
    hidden_activations.resize((unsigned int) hidden_neuron_count,0);
    output_activations.resize((unsigned int) output_neuron_count,0);

}

void NeuralNetwork::feed_forward(const std::vector<double> &input){
	// error check: match input size to size of input neurons
	if (input_neurons.size() != input.size())
	{
		std::cerr << "NeuralNetwork::feed_forward: input size does not match! Input neuron count (w/o bias): "  << input_neuron_count << " input vector size: " <<input.size() << ". Exiting..." << std::endl;
		exit(1);
	}

	// fill all neuron values with 0
	std::fill(input_neurons.begin(),input_neurons.end(),0);
	std::fill(hidden_neurons.begin(), hidden_neurons.end(),0);
	std::fill(output_neurons.begin(), output_neurons.end(),0);

	// fill in input neurons
	for (unsigned int i = 0; i <input_neurons.size(); ++i)
	{
		input_activations[i] = input[i];
	}

	update_activations();
}

void NeuralNetwork::update_activations(){
    for( unsigned int i = 0; i < hidden_neurons.size(); ++i)
    {
        hidden_neurons[i] = -1* weights_in2hid[i][0];
        for(unsigned int j = 0; j < input_neurons.size(); ++j )
        {
            hidden_neurons[i] += weights_in2hid[i][j+1] * input_activations[j];
        }
        hidden_activations[i] = SIGMOID(hidden_neurons[i]);
    }

    for( unsigned int i = 0; i < output_neurons.size(); ++i)
    {
        output_neurons[i] = -1 * weights_hid2out[i][0];
        for(unsigned int j = 0; j < hidden_neurons.size(); ++j)
        {
            output_neurons[i] += weights_hid2out[i][j+1]*hidden_activations[j];
        }
        output_activations[i] = SIGMOID(output_neurons[i]);
    }
}

// write weights of neural net into filestream
void NeuralNetwork::save_weights(std::ofstream *out){
	*out << input_neuron_count << " " << hidden_neuron_count << " " << output_neuron_count << std::endl;

	for (int i = 0; i < hidden_neuron_count; ++i)
	{
		for (int j = 0; j < input_neuron_count +1; ++j)
		{
			*out << std::setprecision(3) << std::fixed << weights_in2hid[i][j];
		    if ( j != input_neuron_count)
                *out << " ";
        }
		*out << std::endl;
	}

	for (int i = 0; i < output_neuron_count; ++i)
	{
		for (int j = 0; j < hidden_neuron_count + 1; ++j)
		{
			*out << std::setprecision(3) << std::fixed << weights_hid2out[i][j];
            if ( j != hidden_neuron_count)
                *out << " ";
		}
		*out << std::endl;
	}

}
