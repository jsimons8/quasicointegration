//
// Created by Jerome Simons on 05/03/2023.
//

#ifndef QUASICOINTEGRATION_MAKEPARAMS_H
#define QUASICOINTEGRATION_MAKEPARAMS_H


#include <Eigen/Dense>

class MakeParams {

public:
    Eigen::MatrixXd R;
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd P;
    Eigen::MatrixXd Diff_mat;
    Eigen::MatrixXd Phi;
    int lag_order;
    Eigen::MatrixXd beta;
    int p;
    Eigen::MatrixXd F;


    MakeParams(const Eigen::Ref<const Eigen::MatrixXd> & R, const Eigen::Ref<const Eigen::MatrixXd> & Lambda);
    Eigen::MatrixXd GenerateF(double root);
    Eigen::MatrixXd GenerateLtilde(double root);
    Eigen::MatrixXd VARSigmaToElliottsOmega(const Eigen::Ref<const Eigen::MatrixXd> & Sigma,bool normalize);
    Eigen::MatrixXd VARSigmaToPredictiveOmega(const Eigen::Ref<const Eigen::MatrixXd> & Sigma,bool normalize);

    Eigen::MatrixXd PredictiveOmegaToVARSigma(const Eigen::Ref<const Eigen::MatrixXd> & Sigma,bool normalize);
    Eigen::MatrixXd PredictiveOmegaToElliotsOmega(const Eigen::Ref<const Eigen::MatrixXd> & Sigma,bool normalize);

    Eigen::MatrixXd ElliottsOmegaToVARSigma(const Eigen::Ref<const Eigen::MatrixXd> & Sigma,bool normalize);



};


#endif //QUASICOINTEGRATION_MAKEPARAMS_H
