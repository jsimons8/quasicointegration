//
// Created by Jerome Simons on 03/03/2023.
//

#ifndef QUASICOINTEGRATION_READDATASET_H
#define QUASICOINTEGRATION_READDATASET_H
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;
class ReadDataSet {

public:

    ReadDataSet(string & data_path,string & column1,string & column2);
    void ReadInWeightsCritVal(const string & weightspath,
                         const string & critical_value_path);
    ReadDataSet(const string & path,int type);

    void ReadInWeights(string weightspath);


    MatrixXd dat;
    double critical_value;
    MatrixXd weights;


};


#endif //QUASICOINTEGRATION_READDATASET_H
