#include "neural.h"

#define hiddenLength 16
#define outLength 10

int main() {

    char num = '0';

while(num >= '0' && num <= '9') {

    std::string filename = std::string(num) + ".bmp";

    arma::vec target = arma::vec(10, arma::fill::zeros);
    target(num - '0') = 1.00;

    arma::mat firstLayer = toGrey(filename);
    arma::mat layerOneWeights = fileRead("layerOneWeights");
    arma::mat layerTwoWeights = fileRead("layerTwoWeights");
    arma::mat layerThreeWeights = fileRead("layerThreeWeights");

    if(num == '0') initWeights(firstLayer, layerOneWeights, layerTwoWeights, layerThreeWeights);

    fileWrite(firstLayer, "firstLayer");
    fileWrite(layerOneWeights, "layerOneWeights");
    fileWrite(layerTwoWeights, "layerTwoWeights");
    fileWrite(layerThreeWeights, "layerThreeWeights");

    arma::vec outputLayer = findOutput(firstLayer, layerOneWeights, layerTwoWeights, layerThreeWeights);

    arma::vec outputError = error(outputLayer, target);

    std::cout << totalError(outputError) << std::endl;
    ++num;
}
    //std::cout << layerOne << std::endl;
    //std::cout << "blah" << std::endl;
   	//std::cout << firstLayer << std::endl;

    return 0;
}

void initWeights(const arma::mat& firstLayer, arma::mat& layerOneWeights, 
                arma::mat& layerTwoWeights, arma::mat& layerThreeWeights) {

    layerOneWeights = arma::mat(hiddenLength, firstLayer.n_rows * firstLayer.n_cols, arma::fill::randn);
    layerTwoWeights = arma::mat(hiddenLength, hiddenLength, arma::fill::randn);
    layerThreeWeights = arma::mat(outLength, hiddenLength, arma::fill::randn);
}

arma::mat toGrey(std::string filename) {

    std::string file = std::string("Images/") + filename;
    std::string greyfile = std::string("Images/") + "grey" + filename;

    BMP Image;

    Image.ReadFromFile(file.c_str());

    for( int i=0 ; i < Image.TellWidth() ; i++)
    {
        for( int j=0 ; j < Image.TellHeight() ; j++)
        {
            double Temp = (Image(i,j)->Red + Image(i,j)->Green + Image(i,j)->Blue  ) / 3;
            Image(i,j)->Red   = Temp;
            Image(i,j)->Green = Temp;
            Image(i,j)->Blue  = Temp;
        }
    }

    Image.SetBitDepth(8);

    CreateGrayscaleColorTable(Image);

    Image.WriteToFile(greyfile.c_str());

    Image.ReadFromFile(greyfile.c_str());

    arma::Mat<double> firstLayer = arma::Mat<double>(Image.TellHeight(), Image.TellWidth(), arma::fill::zeros);

    for(auto i = 0; i < Image.TellHeight(); ++i) {

        for(auto j = 0; j < Image.TellWidth(); ++j) {

//          std::cout << "(" << i << ", " << j << ")" << "  ";
            firstLayer(i, j) = 1 - Image(i, j)->Red / 255.0;
//          firstLayer(i, j) = Image(i, j)->Red;
        }
    }

    return firstLayer;
}

arma::vec findOutput(const arma::mat& firstLayer, const arma::mat& layerOneWeights, 
                    const arma::mat& layerTwoWeights, const arma::mat& layerThreeWeights) {

    arma::vec layerOne = arma::vec(firstLayer.n_rows * firstLayer.n_cols, arma::fill::zeros);
    int k = 0;

    for(int i = 0; i < firstLayer.n_rows; ++i) {

        for(int j = 0; j < firstLayer.n_cols; ++j) {

            layerOne(k) = firstLayer(i, j);
            ++k;
        }
    }

    arma::vec layerTwo = layerOneWeights * layerOne;
    for(int i = 0; i < layerTwo.size(); ++i) {

        layerTwo(i) = sigmoid(layerTwo(i));
    }

    arma::vec layerThree = layerTwoWeights * layerTwo;
    for(int i = 0; i < layerThree.size(); ++i) {

        layerThree(i) = sigmoid(layerThree(i));
    }

    arma::vec outputLayer = layerThreeWeights * layerThree;

    for(int i = 0; i < outputLayer.size(); ++i) {

        outputLayer(i) = sigmoid(outputLayer(i));
    }

    return outputLayer;
}

arma::vec error(const arma::vec& column, const arma::vec& target) {

    arma::vec outputError = arma::vec(column.size(), arma::fill::zeros);

    for(int i = 0; i < outputError.size(); ++i) {

        outputError(i) = 0.5 * (column(i) - target(i)) * (column(i) - target(i));
    }

    return outputError;
}

double totalError(const arma::vec& column) {

    double result = 0;

    for(int i = 0; i < column.size(); ++i) {

        result += column(i);
    }

    return result;
}
void fileWrite(const arma::Mat<double>& matrix, std::string layer) {

    std::ofstream fout;

    fout.open(layer + ".out");

    fout << matrix.n_rows << " " << matrix.n_cols << std::endl;

    for(int i = 0; i < matrix.n_rows; ++i) {

        for(int j = 0; j < matrix.n_cols; ++j) {

            fout << matrix(i, j) << " ";
        }
        fout << std::endl;
    }
}

arma::Mat<double> fileRead(std::string layer) {

    std::ifstream fin;
    fin.open(layer + ".out");
    int r, c;
    double ele;
    fin >> r >> c;

    arma::Mat<double> matrix = arma::Mat<double>(r, c, arma::fill::zeros);

    for(auto i = 0; i < r; ++i) {

        for(auto j = 0; j < c; j++) {

            fin >> ele;
            matrix(i, j) = ele;
        }
    }

    return matrix;
}

double sigmoid(const double& x) {

    double result;

    result = 1 / (1 + exp(-1 * x));

    return result;
}