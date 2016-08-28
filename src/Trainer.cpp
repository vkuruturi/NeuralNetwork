#include <vector>
#include "Trainer.h"
#include <string>
#include <iostream>
#include <sstream>

void Trainer::import_neural_net(std::ifstream *in)
{
	net = new NeuralNetwork(in);
	inputs = net->input_neuron_count;
	outputs = net->output_neuron_count;
}

void Trainer::export_neural_net(std::ofstream *out)
{
	net->save_weights(out);
}

void Trainer::back_prop_learn(std::ifstream *training_file)
{
	int current_epochs = 0;
	std::string s;
	std::vector<double> input_values;
	std::vector<double> output_values;
	std::vector<std::string> v;
	while(std::getline(*training_file,s))
		v.push_back(s);

	while (current_epochs < epochs)
	{
		for(std::vector<std::string>::const_iterator it = v.begin(); it != v.end(); it++)
		{
			int count = 0;
			std::stringstream ss(*it);
			double d;

			while (ss >> d)
			{
				if (count < inputs)
				{
					input_values.push_back(d);
					count++;
				} else
				{
					output_values.push_back(d);
					count++;
				}
			}

			net->feed_forward(&input_values);
			back_prop(&output_values);
		}
		current_epochs++;
	}
}

void Trainer::back_prop(std::vector<double> *out)
{
	std::vector<double> delta_hidden(net->hidden_neuron_count,0);
	std::vector<double> delta_output(net->output_neuron_count,0);

	for(int i = 1; i <= net->output_neuron_count; i++)
		delta_output[i] = (net->output_neurons[i]) * (1-net->output_neurons[i]) * ((*out)[i-1] - (net->output_neurons)[i]);

	for(int i = 0; i <= net->hidden_neuron_count; i++){
		double weight_delta = 0;
		for(int j = 1; j <= net->output_neuron_count; j++){
			weight_delta += (net->weights_hid2out[i][j] * delta_output[j]);
		}

		delta_hidden[i] = (net->hidden_neurons[i]) * (1 - net->hidden_neurons[i] * weight_delta);
	}

	for(int i = 0; i <= net->hidden_neuron_count; i++)
		for(int j = 0; j <= net->output_neuron_count; j++)
			net->weights_hid2out[i][j] = net->weights_hid2out[i][j] + learning_rate * net->hidden_neurons[i] * delta_output[j];

	for(int i = 0; i <= net->input_neuron_count; i++)
		for(int j = 0; j <= net->hidden_neuron_count; j++)
			net->weights_in2hid[i][j] = net->weights_in2hid[i][j] + learning_rate * net->input_neurons[i] * delta_hidden[j];
}