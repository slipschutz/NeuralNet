
CXX=g++
CFLAGS= -g -std=c++11 -I/Users/samlipschutz/Codes/NeuralNet/Eigen/ -O3

EXE=main
OBJS=main.o Network.o

$(EXE) : main.o Network.o
	$(CXX) $(CFLAGS) $(OBJS) -o $@

%.o : %.cc 
	$(CXX) -c $(CFLAGS) -fPIC $< -o $@

clean:
	rm -f $(OBJS) $(EXE)