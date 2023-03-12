//
// Created by Jerome Simons on 05/03/2023.
//

#include "MakeParams.h"
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;




MakeParams::MakeParams(const Eigen::Ref<const Eigen::MatrixXd> &R, const Eigen::Ref<const Eigen::MatrixXd> &Lambda):R(R),
                                                                                                                    Lambda(Lambda) {

    this->p = R.rows();
    double D = R(1,0);
    this->P = MatrixXd(p,p);
    this->P <<  1,0,
            D,1;

    double rho = Lambda(0,0);
    Eigen::MatrixXd M_e(R.rows(),R.rows());
    M_e << rho,0,
            D*rho,0;
    this->Diff_mat = (Eigen::MatrixXd::Identity(R.rows(),R.rows()) - M_e);
    this->lag_order = R.cols() / R.rows(); //R is p x pk

    const int q=1;
    this->beta = Eigen::MatrixXd(p,p-q);
    this->beta(0,0) = 1. * R(1,0);
    this->beta(1,0) = -1.;

    //assemble DGP parts
    int lag_order = R.cols() / R.rows(); //R is p x pk
    int p = R.rows();
    Eigen::MatrixXd R_tilde = Eigen::MatrixXd::Zero(Lambda.rows(),Lambda.cols());
    int pow = 0;
    for (int e = lag_order; e > 0; e--) {
        pow = e - 1;
        R_tilde.block((lag_order - e)*p, 0, p, lag_order*p) = R * Lambda.pow(pow);
    }

    this->F = R_tilde * Lambda * R_tilde.inverse();

    MatrixXd Asum = MatrixXd::Zero(p,p);
    for(int i=0;i<lag_order;i++) Asum += F.block(0,i*p,p,p).eval();
    this->Phi = Eigen::MatrixXd::Identity(p,p)-Asum;




}

Eigen::MatrixXd MakeParams::GenerateF(double root) {


    //prepare parameters
    this->Lambda(0,0) = root;
    //assemble DGP parts
    int lag_order = R.cols() / R.rows(); //R is p x pk
    int p = R.rows();
    Eigen::MatrixXd R_tilde = Eigen::MatrixXd::Zero(Lambda.rows(),Lambda.cols());
    int pow = 0;
    for (int e = lag_order; e > 0; e--) {
        pow = e - 1;
        R_tilde.block((lag_order - e)*p, 0, p, lag_order*p) = R * Lambda.pow(pow);
    }

    this->F = R_tilde * Lambda * R_tilde.inverse();

    return this->F;
}

Eigen::MatrixXd MakeParams::GenerateLtilde(double root) {
    //prepare parameters
    this->Lambda(0,0) = root;
    //assemble DGP parts
    int lag_order = R.cols() / R.rows(); //R is p x pk
    int p = R.rows();
    Eigen::MatrixXd R_tilde = Eigen::MatrixXd::Zero(Lambda.rows(),Lambda.cols());
    int pow = 0;
    for (int e = lag_order; e > 0; e--) {
        pow = e - 1;
        R_tilde.block((lag_order - e)*p, 0, p, lag_order*p) = R * Lambda.pow(pow);
    }
    return R_tilde.inverse().transpose();
}

Eigen::MatrixXd MakeParams::VARSigmaToPredictiveOmega(const Eigen::Ref<const Eigen::MatrixXd> &Sigma, bool normalize) {
    MatrixXd L_tilde = this->GenerateLtilde(this->Lambda(0,0));
    int q = 1;

    //make predictive regression parameters
    MatrixXd K(p,p);
    MatrixXd L = L_tilde.bottomRows(p);
    MatrixXd L_st = L.rightCols(lag_order*p-q);
    MatrixXd L_lu = L.leftCols(q);
    K.topRows(1) = beta.transpose() * R.rightCols(p*lag_order-q) * (MatrixXd::Identity(p*lag_order-q,p*lag_order-q) - Lambda.block(q,q,p*lag_order - q,p*lag_order - q)).inverse() * L_st.transpose();
    MatrixXd L_lu_n = L_lu;
    L_lu_n(0,0) = 1.;
    L_lu_n(1,0) = L_lu(1,0)/L_lu(0,0);
    K.bottomRows(1) = L_lu_n.transpose();
    //std::cout << "K is\n" << K << std::endl;
    //std::cout << "L_lu is " << 1 << " and " << L_lu(1,0)/L_lu(0,0) << std::endl;

    MatrixXd Omega_predictive = K * Sigma * (K.transpose()) ;
    MatrixXd Omega_predictive_n = Omega_predictive / sqrt((Omega_predictive * Omega_predictive.transpose()).trace());

    if(normalize) return Omega_predictive_n; else return Omega_predictive;
}

Eigen::MatrixXd MakeParams::VARSigmaToElliottsOmega(const Eigen::Ref<const Eigen::MatrixXd> &Sigma,bool normalize) {

    Eigen::MatrixXd Elliotts_Omega = P.inverse() * Diff_mat * Phi.inverse() * Sigma * Phi.inverse().transpose() * Diff_mat.transpose() * P.inverse().transpose();

    if(normalize) {
        return Elliotts_Omega / sqrt((Elliotts_Omega.transpose() *Elliotts_Omega).trace()) ;}
    else {
        return Elliotts_Omega;
    }


}

Eigen::MatrixXd MakeParams::PredictiveOmegaToVARSigma(const Ref<const Eigen::MatrixXd> &Omega_predictive, bool normalize) {
    MatrixXd L_tilde = this->GenerateLtilde(this->Lambda(0,0));
    int q = 1;

    //make predictive regression parameters
    MatrixXd K(p,p);
    MatrixXd L = L_tilde.bottomRows(p);
    MatrixXd L_st = L.rightCols(lag_order*p-q);
    MatrixXd L_lu = L.leftCols(q);
    K.topRows(1) = beta.transpose() * R.rightCols(p*lag_order-q) * (MatrixXd::Identity(p*lag_order-q,p*lag_order-q) - Lambda.block(q,q,p*lag_order - q,p*lag_order - q)).inverse() * L_st.transpose();
    MatrixXd L_lu_n = L_lu;
    L_lu_n(0,0) = 1.;
    L_lu_n(1,0) = L_lu(1,0)/L_lu(0,0);
    K.bottomRows(1) = L_lu_n.transpose();
    //std::cout << "K is\n" << K << std::endl;
    //std::cout << "L_lu is " << 1 << " and " << L_lu(1,0)/L_lu(0,0) << std::endl;

    MatrixXd Sigma = K.inverse() * Omega_predictive * (K.transpose().inverse()) ;
    MatrixXd Sigma_n =  Sigma / sqrt((Sigma * Sigma.transpose()).trace());

    if(normalize) return Sigma_n; else return Sigma;
}

Eigen::MatrixXd MakeParams::PredictiveOmegaToElliotsOmega(const Ref<const Eigen::MatrixXd> &Omega_predictive, bool normalize) {

    MatrixXd Sigma = this->PredictiveOmegaToVARSigma(Omega_predictive,normalize);
    MatrixXd ElliottsOmega = this->VARSigmaToElliottsOmega(Sigma,normalize);
    return ElliottsOmega;

}

Eigen::MatrixXd MakeParams::ElliottsOmegaToVARSigma(const Ref<const Eigen::MatrixXd> &Omega_elliott, bool normalize) {

    MatrixXd Sigma = Phi * Diff_mat.inverse() * P * Omega_elliott * P.transpose() * Diff_mat.inverse().transpose() * Phi.transpose();
    MatrixXd Sigma_n = Sigma / sqrt((Sigma * Sigma.transpose()).trace());

    if(normalize) return Sigma_n; else return Sigma;


}




