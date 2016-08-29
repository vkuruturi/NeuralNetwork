cmake_minimum_required(VERSION 3.6)
project(NeuralNetwork)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(inc)
set(SOURCE_FILES
    inc/NeuralNetwork.h
    inc/Tester.h
    inc/Trainer.h
    src/main.cpp
    src/NeuralNetwork.cpp
    src/Tester.cpp
    src/Trainer.cpp
    test/WDBC/net_init.txt
    test/WDBC/net_trained.txt
    test/WDBC/results.txt
    test/WDBC/test.txt
    test/WDBC/train.txt
    Makefile)

add_executable(NeuralNetwork ${SOURCE_FILES})
