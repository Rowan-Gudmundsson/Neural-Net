#include <iostream>
#include <fstream>
#include <armadillo>
#include <cmath>
#include <string>
#include <iomanip>
#include "EasyBMP.h"

void initWeights(const arma::mat&, arma::mat&, arma::mat&, arma::mat&);
arma::mat toGrey(std::string filename);
arma::vec findOutput(const arma::mat&, const arma::mat&, const arma::mat&, const arma::mat&);
arma::vec error(const arma::vec&, const arma::vec&);
double totalError(const arma::vec&);
void fileWrite(const arma::Mat<double>&, std::string);
arma::Mat<double> fileRead(std::string layer);
double sigmoid(const double&);