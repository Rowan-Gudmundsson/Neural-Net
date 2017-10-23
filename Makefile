all: neural EasyBMP
	g++ -Wall -o neural neural.o EasyBMP.o -larmadillo -std=c++11

EasyBMP: EasyBMP.cpp EasyBMP.h
	g++ -c EasyBMP.cpp -std=c++11

neural: neural.cpp neural.h
	g++ -c neural.cpp  -std=c++11
