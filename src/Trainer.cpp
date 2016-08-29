#include <vector>
#include "Trainer.h"
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

#define SIGMOID(x)  (1/(1+exp(-x)))
#define SIGDERIV(x) (SIGMOID(x) * (1-SIGMOID(x)))


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

    std::vector<std::string> v;
    std::vector<double> input_values;
    std::vector<bool> output_values;

    int current_epochs = 0;
	std::string s;

	getline(*training_file,s);
    while(getline(*training_file,s))
    {
        v.push_back(s);
    }

    while (current_epochs < epochs)
	{
		for(unsigned int i = 0; i < v.size(); i++)
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
					output_values.push_back((bool) d);
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


    for(int j = 0; j < net->output_neuron_count; ++j)
    {
        double sum = -1 * net->weights_hid2out[j][0];
        for(int i = 0; i < net->hidden_neuron_count; ++i)
        {
            sum += (net->hidden_activations[i] * net->weights_hid2out[j][i+1]);
        }
        delta_output[j] = SIGDERIV(sum)*((*out)[j] - SIGMOID(sum));
    }

	for(int i = 0; i < net->hidden_neuron_count; ++i)
    {
        double sum = 0;
		for(int j = 0; j < net->output_neuron_count; ++j)
        {
			sum += (net->weights_hid2out[j][i+1] * delta_output[j]);
		}

		delta_hidden[i] = SIGDERIV(net->hidden_neurons[i]) * sum;

        net->weights_in2hid[i][0] += learning_rate * -1 * delta_hidden[i];
        for(int j = 0; j < net->input_neuron_count; ++j)
        {
            net->weights_in2hid[i][j+1] += learning_rate * net->input_activations[j] * delta_hidden[i];
        }
	}

    for(int j = 0; j < net->output_neuron_count; ++j)
    {
        net->weights_hid2out[j][0] += -1* learning_rate * delta_output[j];
        for(int i = 0; i < net->hidden_neuron_count; ++i)
        {
            net->weights_hid2out[j][i+1] += learning_rate * net->hidden_activations[i] * delta_output[j];
        }
    }

}