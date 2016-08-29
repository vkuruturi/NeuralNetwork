#include "NeuralNetwork.h"
#include "Trainer.h"
#include "Tester.h"

#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sstream>

int getnum(void)
{
    int x = 0;
    bool loop = true;

    while(loop)
    {
        std::string s;
        std::getline(std::cin, s);
        std::stringstream stream(s);
        
        if(stream >> x)
        {
            loop = false;
            continue; 
        }

        std::cout << "Invalid input! Please input a number!" << std::endl;
    }

    return x;
}

void train(void)
{
	Trainer t;
	int i;
	double d;
	std::string s;
	std::ifstream net_file, training_file;
	std::ofstream outfile;
	std::cout << "Specify input file to load the initial neural network from: ";
	std::cin >> s;

	//s = "../test/grades/net_init.txt";
	net_file.open(s.c_str());
    
    while (!net_file.good())
    {
    	std::cout << "Unable to open file! Try again: ";
        std::cin >> s;
        net_file.open(s.c_str());
    }
    t.import_neural_net(&net_file);

	std::cout << "Enter the number of epochs training should go on for: ";
	std::cin >> i;
	//i = 100;
	t.set_epochs(i);
	std::cout << "Enter the learning rate: ";
	std::cin >> d;
	//d = 0.05;
	t.set_learning_rate(d);

	std::cout << "Specify the input file for training the neural network: ";
	std::cin >> s;
    //s = "../test/grades/train.txt";
	training_file.open(s.c_str());
    
    while (!training_file.good())
    {
    	std::cout << "Unable to open file! Try again: ";
        std::cin >> s;
        training_file.open(s.c_str());
    }

    std::cout << "Starting training..." << std::endl;
    training_file >> i;
    t.set_examples(i);
    training_file >> i;
    t.set_inputs(i);
    training_file >> i;
    t.set_outputs(i);

    t.back_prop_learn(&training_file);

    std::cout << "Finished training!" << std::endl << std::endl;

    std::cout << "Enter output file name to save the neural network into: ";
    std::cin >> s;
    //s = "net_output.txt";
    outfile.open(s.c_str());
    while(!outfile.good())
    {
    	std::cout << "Unable to open file! Try again: ";
    	std::cin >> s;
    	outfile.open(s.c_str());
    }

    std::cout << "Exporting neural network...";
    t.export_neural_net(&outfile);

    std::cout << "Training completed." << std::endl;
}

void test()
{
    std::string s;
    std::ifstream net_file, test_file;
    std::ofstream outfile;
    std::cout << "Specify input file to load the trained neural network from: ";
    std::cin >> s;

    //s = "../test/grades/net_trained.txt";
    net_file.open(s.c_str());

    while (!net_file.good())
    {
        std::cout << "Unable to open file! Try again: ";
        std::cin >> s;
        net_file.open(s.c_str());
    }
    Tester t(&net_file);

    std:: cout << "Specify input file to load the test data from: ";
    std::cin >> s;
    //s = "../test/grades/test.txt";
    test_file.open(s.c_str());

    while(!test_file.good())
    {
        std::cout << "Unable to open file! Try again: ";
        std::cin >> s;
        test_file.open(s.c_str());
    }
    std::cout << std::endl;
    std::cout << "Starting test..." << std::endl;
    t.test_network(&test_file);
    std::cout << "Test finished!" << std::endl;
    std::cout << std::endl;

    std::cout << "Specify output file to write results to: ";
    std::cin >> s;
    //s = "results.txt";
    outfile.open(s.c_str());
    while(!outfile.good())
    {
        std::cout << "Unable to open file! Try again: ";
        std::cin >> s;
        outfile.open(s.c_str());
    }
    t.output_metrics(&outfile);
    std::cout << "Testing completed." << std::endl;

}

int main(int argc, char** argv)
{
	std::cout << std::endl;

	std::cout << "1.  Train a neural network" << std::endl;
	std::cout << "2.  Test a neural network" << std::endl;

	while(1){
		int i = getnum();
		if ( i == 1){
			train();
			break;
		}
		else if (i == 2){
			test();
			break;
		}
		else
			std::cout << "Invalid input!  Try again" << std::endl;
	}

}