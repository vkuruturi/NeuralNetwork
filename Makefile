CXX=g++
CXXFLAGS=-std=c++11 -Wall -Iinc/
BIN=NeuralNet.exe
CXX_DEBUG_FLAGS=-g

SRCDIR= src
SRC=$(wildcard $(SRCDIR)/*.cpp)
OBJ=$(SRC:%.cpp=%.o)

all: $(OBJ)
	$(CXX) -o $(BIN) $+ 

%.o: %.c %.h
	$(CXX) $@ -c $< 


.PHONY: debug
debug: CXXFLAGS+=$(CXX_DEBUG_FLAGS)
debug: all

.PHONY: clean
clean: 
	rm -f $(SRCDIR)/*.o $(BIN)  *~
