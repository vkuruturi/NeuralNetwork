#include <vector>
#include "Trainer.h"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

	std::vector<std::string> v;
	std::vector<double> input_values;
	std::vector<double> output_values;

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

	int i = 0;
	std::cout << "initial size: " << v.size();
    while(getline(*training_file,s))
    {
        v.push_back(s);
        //std::cout << v[i] << std::endl;
        i++;
    }

    for(int i = 0; i < v.size(); i++)
    {
    //	std::cout << v[i] << std::endl;
    }
epochs=1;
	std::cout << "example count: " << v.size() << std::endl;
	while (current_epochs < epochs)
	{
		for(int i = 0; i < v.size(); i++)
		{
			std::cout << v[i] << std::endl;
			int count = 0;
			std::stringstream ss(v[i]);
			double d;
			std::cout << "COUNT: " << count << std::endl;
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
			std::cout << input_values.size() << std::endl;
			net->feed_forward(input_values);
			//back_prop(&output_values); 
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