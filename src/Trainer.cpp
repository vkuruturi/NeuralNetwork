#include <vector>
#include "Trainer.h"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

#define SIGMOID(x)  (1/(1+exp(-x)))
#define SIGDERIV(x) (SIGMOID(x) * (1-SIGMOID(x)))

	std::vector<std::string> v;
	std::vector<double> input_values;
	std::vector<bool> output_values;

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
    std::cout << "Learning rate: " << learning_rate << " epochs: " << epochs << std::endl;
	int current_epochs = 0;
	std::string s;

	getline(*training_file,s);
    while(getline(*training_file,s))
    {
        v.push_back(s);
    }

    while (current_epochs < epochs)
	{
		for(int i = 0; i < v.size(); i++)
		{
			int count = 0;
			std::stringstream ss(v[i]);
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
			net->feed_forward(input_values);
			back_prop(&output_values); 

			input_values.clear();
			output_values.clear();

		}
		current_epochs++;
	}
}

void Trainer::back_prop(std::vector<bool> *out)
{
	std::vector<double> delta_hidden((unsigned int) net->hidden_neuron_count,0);
	std::vector<double> delta_output((unsigned int) net->output_neuron_count,0);

	for(int i = 0; i < net->output_neuron_count; ++i)
		delta_output[i] = SIGDERIV(net->output_neurons[i]) * ((*out)[i] - net->output_activations[i]);

	for(int i = 0; i < net->hidden_neuron_count; ++i)
    {
		for(int j = 0; j < net->output_neuron_count; ++j)
        {
			delta_hidden[i] += (net->weights_hid2out[j][i+1] * delta_output[j]);
		}

		delta_hidden[i] = SIGDERIV(net->hidden_neurons[i]);
	}

	for(int i = 0; i < net->output_neuron_count; ++i)
    {
        for (int j = 0; j < net->hidden_neuron_count; ++j)
        {
            net->weights_hid2out[i][j + 1] =
                    net->weights_hid2out[i][j + 1] + (learning_rate * net->hidden_activations[j] * delta_output[i]);
        }
        net->weights_hid2out[i][0] = net->weights_hid2out[i][0] + learning_rate * -1 * delta_output[i];
    }

	for(int i = 0; i < net->hidden_neuron_count; ++i)
    {
        for (int j = 0; j < net->input_neuron_count; ++j)
        {
            //std::cout << i << " " << j << std::endl;
            net->weights_in2hid[i][j + 1] =
                    net->weights_in2hid[i][j + 1] + (learning_rate * net->input_activations[j] * delta_hidden[i]);
        }
        net->weights_in2hid[i][0] = net->weights_in2hid[i][0] + learning_rate * -1 * delta_hidden[i];
    }
}