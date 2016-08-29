#include "Tester.h"
#include "NeuralNetwork.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>


Tester::Tester(std::ifstream *in)
{
	net = new NeuralNetwork(in);
}

void Tester::test_network(std::ifstream *in)
{
	std::string line;

	std::stringstream ss;
	std::vector<std::string> data;
	int samples, inputs, outputs;
	samples = inputs = outputs = 0;

	std::getline(*in, line);
	ss.str(line);
	ss >> samples >> inputs >> outputs;

	std::vector<bool> v(outputs);
	classified.resize(samples,v);
	expected_output.resize(samples, v);
    //std::cout << "expected output size: " << expected_output.size() << " " << expected_output[0].size() << std::endl;

	while(std::getline(*in,line))
		data.push_back(line);


	for( unsigned int i = 0; i < data.size(); ++i)
	{
		input.clear();
		std::stringstream ss(data[i]);
		//std::cout << data[i] << std::endl;
		double d = 0;
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
				expected_output[i][count - net->input_neuron_count] = (bool) d;
				count++;
			}
		}
		//std::cout << i << std::endl;
		net->feed_forward(input);
		for(int j = 0; j < outputs; ++j)
		{
			classified[i][j] = (net->output_activations[j] >= 0.5 ? true : false);
		}
	}
}


void Tester::output_metrics(std::ofstream * outfile)
{
	double total_A = 0,total_B = 0,total_C = 0,total_D = 0;

	double micro_accuracy = 0, micro_precision = 0, micro_recall = 0, micro_F1 = 0;
	double macro_accruacy = 0, macro_precision = 0, macro_recall = 0, macro_F1 = 0;

	for( int i = 0; i < net->output_neuron_count; ++i)
	{
		double A,B,C,D;
		A = B = C = D = 0;

		for(unsigned int j = 0; j < classified.size(); j++)
		{
			if(expected_output[j][i] && classified[j][i])
				A++;
			if(!expected_output[j][i] && classified[j][i])
				B++;
			if(expected_output[j][i] && !classified[j][i])
				C++;
			if(!expected_output[j][i] && !classified[j][i])
				D++;

		}

		double accuracy = (A + D) / (A + B + C + D);
		double precision = A / (A+B);
		double recall = A / (A+C);
		double f1 = (2* precision*recall) / (precision + recall);

		*outfile << std::setprecision(0) << std::fixed << A << " " << B << " " << C << " " << D
				 << " " << std::setprecision(3) << std::fixed << accuracy << " "
				 << precision << " " << recall << " " << f1 << std::endl;

		total_A += A;
		total_B += B;
		total_C += C;
		total_D += D;

		macro_accruacy 	+= accuracy/net->output_neuron_count;
		macro_precision += precision/net->output_neuron_count;
		macro_recall 	+= recall/net->output_neuron_count;
		macro_F1		+= f1/net->output_neuron_count;
	}
    //std::cout << total_A << " " << total_B << " " << total_C << " " << total_D << std::endl;
	micro_accuracy	= (total_A + total_D) / (total_A + total_B + total_C + total_D);
	micro_precision = total_A / (total_A + total_B);
	micro_recall	= total_A / (total_A + total_C);
	micro_F1		= (2 * micro_precision * micro_recall) / (micro_precision + micro_recall);

	macro_F1 		= (2 * macro_precision * macro_recall) / (macro_precision + macro_recall);

	*outfile << std::setprecision(3) << std::fixed << micro_accuracy << " " << micro_precision << " " << micro_recall << " " << micro_F1 << std::endl;
	*outfile << std::setprecision(3) << std::fixed << macro_accruacy << " " << macro_precision << " " << macro_recall << " " << macro_F1 << std::endl;
}


