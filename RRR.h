//
// Created by Jerome Simons on 03/03/2023.
//

#ifndef QUASICOINTEGRATION_RRR_H
#define QUASICOINTEGRATION_RRR_H


#include <Eigen/Dense>

class RRR {

public:
    int fitted_lag_order;
    int T;
    double ll;
    double critical_value;
    Eigen::MatrixXd dat;
    Eigen::VectorXd Dvec_2;
    RRR(const Eigen::Ref<const Eigen::MatrixXd> & data_matrix,int fitted_lag_order);
    void GetStart(double nu, bool intercept);
    bool restricted;
    double LongRunCR();

    void MakeOLSEstimates();




    Eigen::MatrixXd embed(const Eigen::Ref<const Eigen::MatrixXd> &dat_in,int fitted_lag_order);
    //remember to read them backwards and there are no const references, only references to constants.
    int operator()(const Eigen::VectorXd&, Eigen::VectorXd&) const;

//outcomes to be set

    Eigen::MatrixXd S11;
    Eigen::MatrixXd S10;
    Eigen::MatrixXd S00;
    Eigen::MatrixXd R_lu;
    Eigen::MatrixXd alpha_perp;
    Eigen::MatrixXd alpha_perp2;
    Eigen::MatrixXd beta;
    Eigen::MatrixXd beta_perp;
    Eigen::MatrixXd alpha;
    Eigen::MatrixXd residuals;

    Eigen::MatrixXd F;
    Eigen::MatrixXd Sigma;

};



#endif //QUASICOINTEGRATION_RRR_H
