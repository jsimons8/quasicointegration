//
// Created by Jerome Simons on 06/03/2023.
//

#ifndef QUASICOINTEGRATION_GENERATEDATASET_H
#define QUASICOINTEGRATION_GENERATEDATASET_H

#include <Eigen/Dense>

class GenerateDataSet {

public:
    int p;
    int k_dgp;
    int T;
    Eigen::MatrixXd F;
    GenerateDataSet(int,int,int, const Eigen::Ref<const Eigen::MatrixXd> &);

    Eigen::MatrixXd Generate(const Eigen::Ref<const Eigen::MatrixXd>& error_mat);

};

#endif //QUASICOINTEGRATION_GENERATEDATASET_H
