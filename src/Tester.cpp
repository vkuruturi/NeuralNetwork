#include "Tester.h"
#include "NeuralNetwork.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>

#define ROUND(x) ( (x >= 0.5) ? true : false )

Tester::Tester(std::ifstream *in)
{
	net = new NeuralNetwork(in);
	for(int i = 0; i < 4; i++)
	{
		macro_metrics[i] = 0;
		micro_metrics[i] = 0;
	}
}

void Tester::test_network(std::ifstream *in)
{
	std::string line;
	std::vector<double> input;
	std::vector<bool> expected_output;
	*in >> line;

	std::vector<std::string> data;
	while(std::getline(*in,line))
		data.push_back(line);

	for(std::vector<std::string>::const_iterator it = data.begin(); it != data.end(); it++)
	{
		input.clear();
		expected_output.clear();
		std::stringstream ss(line);
		double d;
		int count = 0;

		while( ss >> d)
		{
			if (count < net->input_neuron_count)
			{
				input.push_back(d);
				count++;
			}
			else
			{
				expected_output.push_back(d);
				count++;
			}
		}

		net->feed_forward(&input);
		compare_outputs(&expected_output);
	}
	calculate_metrics();
}

void Tester::compare_outputs(std::vector<bool> *expected_output)
{
	std::vector<double>::iterator neuron_it = (net->output_neurons.begin())++;
	std::vector<bool>::iterator expected_it = expected_output->begin();
	std::vector<std::vector<double>>::iterator metric_it = metrics.begin();
	for(; neuron_it != net->output_neurons.end(); neuron_it++, expected_it++, metric_it++)
	{
		if (ROUND(*neuron_it))
		{
			if(*expected_it)
				((*metric_it)[0])++;
			else ((*metric_it)[1])++;
		}
		else
		{
			if(*expected_it)
				((*metric_it)[2])++;
			else ((*metric_it)[3])++;
		}
	}
}

void Tester::calculate_metrics()
{

	for(std::vector<std::vector<double>>::iterator it = metrics.begin(); it != metrics.end(); it++)
	{
		// overall accuracy for the category
		(*it)[4] = ((*it)[0] + (*it)[3]) / ((*it)[0] + (*it)[1] + (*it)[2] + (*it)[3]);

		// precision
		(*it)[5] = ((*it)[0]) / ((*it)[0] + (*it)[1]);

		// recall
		(*it)[6] = ((*it)[0]) / ((*it)[0] + (*it)[2]);

		//F1
		(*it)[7] = (2 * (*it)[6] * (*it)[5]) / ((*it)[5] + (*it)[6]);

		for (int i = 0; i < 4; i++)
		{
			micro_metrics[i] += (*it)[i];
			macro_metrics[i] += (*it)[i+4];
		}
	}

	double accuracy, precision, recall;

	accuracy  = (micro_metrics[0] + micro_metrics[3]) / (micro_metrics[0] + micro_metrics[1] + micro_metrics[2] + micro_metrics[3]);
	precision = micro_metrics[0] / (micro_metrics[0] + micro_metrics[1]);
	recall    = micro_metrics[0] / (micro_metrics[0] + micro_metrics[2]);
	micro_metrics[3] = (2 * micro_metrics[1] * micro_metrics[2]) / (micro_metrics[1] + micro_metrics[2]);

	for(int i = 0; i < 4; i++)
		macro_metrics[i] /= net->output_neuron_count;

	macro_metrics[3] = (2 * macro_metrics[1] * macro_metrics[2]) / (macro_metrics[1] + macro_metrics[2]);
}

void Tester::output_metrics(std::ofstream * metrics_file)
{

	for( auto it = metrics.begin(); it != metrics.end(); it++)
	{
		for ( int i = 0; i < it->size(); i++)
		{
			if ( i < (*it).size() / 2)
				*metrics_file << std::setprecision(0) << std::fixed << (*it)[i];
			else
				*metrics_file << std::setprecision(3) << std::fixed << (*it)[i];
			if ( i < it->size())
				*metrics_file << " ";
			else 
				*metrics_file << std::endl;
		}
	}

	for(int i = 0; i < 4; i++)
	{
		*metrics_file << micro_metrics[i];
		if ( i < 3)
			*metrics_file << " ";
		else *metrics_file << std::endl;
	}

	for(int i = 0; i < 4; i++)
	{
		*metrics_file << macro_metrics[i];
		if ( i < 3)
			*metrics_file << " ";
		else *metrics_file << std::endl;
	}
}