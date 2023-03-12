//
// Created by Jerome Simons on 03/03/2023.
//

#include <iostream>
#include "RRR.h"

using namespace Eigen;

RRR::RRR(const Eigen::Ref<const Eigen::MatrixXd> &data_matrix, int fitted_lag_order):dat(data_matrix),
                                                                                     fitted_lag_order(fitted_lag_order){
    this->T = this->dat.rows();
    this->ll = 0.;
    this->critical_value=0.;
    restricted = true;
}


double RRR::LongRunCR(){


    const int p = 2;
    int lag_order = 3;

    MatrixXd X_luet2 = MatrixXd::Zero(p*lag_order, T - lag_order);

    MatrixXd persistent_series = this->residuals * this->alpha_perp;
    MatrixXd equilibrium_errors =   this->dat.bottomRows(this->T-fitted_lag_order) * this->beta;
    MatrixXd persistent_eq_error_series(this->T-fitted_lag_order,2);
    persistent_eq_error_series << equilibrium_errors, persistent_series;
    int lags = 6;//pow(this->T,1./3.);
    const int numvars = persistent_eq_error_series.cols();
    const int N = persistent_eq_error_series.rows();

    MatrixXd X_luet = MatrixXd::Zero(p*lags, N - lags);
    MatrixXd Y_luet = persistent_eq_error_series.bottomRows(N - lags).transpose();

    for (int i = lags; i > 0; i--)
    {
        X_luet.block((lags - i)*p, 0, p, N - lags) = persistent_eq_error_series.block(i - 1, 0, N - lags, p).transpose();
    }

    //time to make the long-run covariance A implied here is A'
    MatrixXd Atemp = ( (X_luet * X_luet.transpose()).ldlt().solve(X_luet * Y_luet.transpose() )).transpose();


    MatrixXd res = Y_luet - Atemp * X_luet;
    MatrixXd SR_cov = res * res.transpose() / static_cast<double>(this->T);
    //sum the columns
    MatrixXd Asum(p,p);
    for(int i=0;i<lags;i++) Asum += Atemp.block(0,i*p,p,p).eval();
    MatrixXd F = MatrixXd::Identity(p,p)-Asum;

    MatrixXd LRcov = F.inverse() * SR_cov * F.transpose().inverse();

    double lr = LRcov(0,1) / (sqrt(LRcov(0,0)) * sqrt(LRcov(1,1)));
    //std::cout << "LR corr is\n" << lr << std::endl;


    return lr;




}


void RRR::GetStart(double nu,bool intercept) {

    int p = this->dat.cols();
    int T = this->dat.rows();
    int q = 1;

    //predefine the matrices
    this->S11 = MatrixXd::Zero(p, p);
    this->S10 = MatrixXd::Zero(p, p);
    this->S00 = MatrixXd::Zero(p, p);
    //here we effectively implement the Johansen procedure
    MatrixXd Z(T - fitted_lag_order, fitted_lag_order*p);
    //double nu = Impositions.Lambda_lu(0,0); // this means we are quasi-differencing with the largest root
    //std::cout << "the quasi differencing coefficient is " << nu << std::endl;
    MatrixXd FD = dat.bottomRows(T - 1) - nu * dat.topRows(T - 1);
    Z = this->embed(FD,fitted_lag_order); //T-k+1


    MatrixXd Z0 = Z.leftCols(p);//(T - 1, p);
    //this is the first lag for the short-run implementation of Johansen
    MatrixXd Z1 = dat.block(fitted_lag_order - 1, 0, T - fitted_lag_order, p);
    MatrixXd Z2 = Z.rightCols((fitted_lag_order - 1)*p);
    //begin block to deal with deterministic values
    MatrixXd C(Z2.rows(), Z2.cols()+1); //only intercept henceplus 1 cols
    if(intercept) {

        //Z2 gets additional columns in case of deterministic terms
        MatrixXd C(Z2.rows(), Z2.cols() + 1); //only intercept no dummies
        MatrixXd B(T-fitted_lag_order, 1); //2 for trend and intercept and frequency -1 for the rest

        B.block(0,0,T-fitted_lag_order,1) = MatrixXd::Ones(T-fitted_lag_order,1); //the intercept(s)

        //B.block(0,1,T-fitted_lag_order,1) = VectorXd::LinSpaced(T-fitted_lag_order,1,T-fitted_lag_order); //the trend

        C << B,Z2;


        Z2 = C;

        //std::cout << "C is " << C << std::endl;

    }



    //make crossproduct matrices
    double f = (1 / static_cast<double>(T-fitted_lag_order));
    MatrixXd M00 = f * Z0.transpose() * Z0;
    MatrixXd M11 = f * Z1.transpose() * Z1;
    MatrixXd M10 = f * Z1.transpose() * Z0;
    MatrixXd M22inv((fitted_lag_order - 1)*p, (fitted_lag_order - 1)*p);
    MatrixXd M20((fitted_lag_order - 1)*p, p);

    MatrixXd R0(Z0.rows(),Z0.cols());
    MatrixXd R1(Z1.rows(),Z1.cols());
    MatrixXd M12(p,(fitted_lag_order-1)*p);

    if (fitted_lag_order == 1 && !(intercept)) {
        this->S11 = M11;
        this->S10 = M10;
        this->S00 = M00;
        MatrixXd Psihat(1,1);
        Psihat = MatrixXd::Zero(1,1);
    }
    else  {
        M22inv = (f*Z2.transpose() * Z2).inverse();
        M20 = f * Z2.transpose() * Z0;
        M12 = f * Z1.transpose() * Z2;
        MatrixXd S(p, (fitted_lag_order - 1)*p);
        S = M12 * M22inv;
        this->S11 = M11 - S * M12.transpose();
        this->S10 = M10 - S * M20;
        this->S00 = M00 - M20.transpose() * M22inv * M20;

        //compute residuals:
        R0 = Z0 - Z2 * M22inv.transpose() * M20;
        R1 = Z1 - Z2*M22inv.transpose() * M12.transpose();
    }
    //that was a lot
    SelfAdjointEigenSolver<MatrixXd> rrr(this->S11.rows());
    rrr.compute(this->S11);
    const MatrixXd S11negsqrt = rrr.eigenvectors() * rrr.eigenvalues().array().pow(-.5).matrix().asDiagonal() * rrr.eigenvectors().transpose();
    rrr.compute(S11negsqrt * S10 * S00.inverse() * S10.transpose() * S11negsqrt);
    const MatrixXd V = S11negsqrt * rrr.eigenvectors();
    this->beta = V.rightCols(p-q);
    this->beta_perp = V.leftCols(q);
    const MatrixXd alpha = this->S10.transpose() * beta;
    const MatrixXd alpha_p = this->S00.inverse() * this->S10.transpose() * V.leftCols(q);
    const double log_likelihood = this->S00.determinant() * (1. - rrr.eigenvalues().tail(p-q).array()).prod();
    this->alpha_perp2 = alpha_p;



    MatrixXd Pi = alpha * this->beta.transpose();

    //time for residuals:
    if(fitted_lag_order == 1 && !(intercept))
    {
        this->residuals = Z0 -  Z1 * Pi.transpose();
    }
    else
    {
        this->residuals = R0 - R1 * Pi.transpose();
    }



    MatrixXd L(S11.llt().matrixL());
    L = L.inverse().eval();
    MatrixXd quadform = L * S10 * S00.inverse() * S10.transpose() * L.transpose();
    //On to eigendecompositions
    SelfAdjointEigenSolver<MatrixXd> eigensolver(quadform);
    if (eigensolver.info() != Success) abort();
    //the deliverables... careful columns are switched, it's in ascending order
    MatrixXd beta_un = L.transpose() * eigensolver.eigenvectors();
    beta_un = beta_un.rightCols(p-q).eval();
    //Make starting values
    FullPivLU<MatrixXd> lu(beta_un.transpose());
    MatrixXd R_lu_start = lu.kernel(); //we need to obtain the nullspace
    //R_lu_0 < -R_lu_0 % *% solve(as.matrix(R_lu_0[1:dgp$q, 1 : dgp$q]))
    //here we are done with the cointegrating matrix
    //normalise
    this->R_lu = (R_lu_start * R_lu_start.block(0, 0, q, q).inverse()).eval();

    //compute the likelihood
    //lambda is a vector of 1-eigenvalues
    VectorXd lambda = VectorXd::Ones(p)- eigensolver.eigenvalues();

    //from here on we need lambda and S00

    //compute the log likelihood in a numerically safe way
    //note that this is the unrestricted
    //this is equation (6.14)
    double product;
    product = 1.0;
    for (int el = p - 1; el > p - q - 1; el--) {
        product *= lambda(el);
    }
    double ll_joh = product * S00.determinant();
    double ll_joh2 = pow(ll_joh, (-0.5) * (1 / f));
    ll_joh = (-0.5) * (1 / f) * log(ll_joh);
    // alt.
    ll_joh2 = log(ll_joh2);

    //set the relevant variables


    //MatrixXd R_lu = R_lu_start;
    this->ll = ll_joh;
    Map<VectorXd> R_lu_vec(this->R_lu.data(),this->R_lu.size());
    VectorXd R_lu_vec2 = R_lu_vec.eval(); //to have the vectorised starting value ready
    //set D_vec too
    MatrixXd  D_start = this->R_lu.block(q, 0, p - q, q); //hand over the starting value
    Map<VectorXd> my_guess_vec(D_start.data(),D_start.size());
    this->Dvec_2 = my_guess_vec.eval();

    //TODO: implement alpha_perp and beta_perp for good measure
    MatrixXd beta_un2 = L.transpose() * eigensolver.eigenvectors();


    this->alpha_perp = this->S00.inverse() * this->S10.transpose() * beta_un2.leftCols(q);
    double coeff = this->alpha_perp(1,0) / this->alpha_perp(0,0);
    this->alpha_perp(0,0) = 1.;
    this->alpha_perp(1,0) = coeff;

}

//when using this function in a different context, be careful, it's designed to do one lag less.
MatrixXd RRR::embed(const Ref<const MatrixXd>& dat_in, int fitted_lag_order)
{
    int T_in = dat_in.rows();
    int p = dat_in.cols();
    MatrixXd embed_mat(T_in - fitted_lag_order + 1, p*fitted_lag_order);
    for (int i = 0; i < fitted_lag_order; i++)
    {
        embed_mat.block(0, i*p, T_in - fitted_lag_order + 1, p) = dat_in.block(fitted_lag_order - i - 1, 0, T_in - fitted_lag_order + 1, p);
    }
    return embed_mat;
}


int RRR::operator()(const Eigen::VectorXd& parameter_vector, Eigen::VectorXd &fvec ) const {
    const VectorXd::Index n = parameter_vector.size(); //important for root solver
    assert(fvec.size()==n);

    int p = this->S00.cols();
    int q = 1;

    Map<const MatrixXd> Dp(parameter_vector.data(), p - q, q);


    /*

    //1. matricise the D guess into an R_lu guess and into b


    MatrixXd G = MatrixXd::Identity(p,p).block(0,0,p,q);
    MatrixXd G_perp = MatrixXd::Identity(p,p).block(0,p-q,p,p-q);

    MatrixXd R_lu = G + G_perp * D;
    assert(R_lu.size()==p*q);

    //2. find b = lu(R_lu.transpose()).kernel
    MatrixXd b(p,p-q);
    FullPivLU<MatrixXd> lu(R_lu.transpose());
    b = lu.kernel();

    */
    MatrixXd b(p,p-q);
    b.block(0,0,p-q,p-q) = MatrixXd::Identity(p-q,p-q);
    b.block(p-q,0,q,p-q) = -Dp.transpose();


    //3. S00=LL'
    MatrixXd L(S00.llt().matrixL());

    //4. (b'S11b)^-1 = K'^-1 K^-1
    MatrixXd K((b.transpose() * S11 * b).llt().matrixL());

    //5. V \defeq = K^-1 b'S10(L')^-1

    MatrixXd V = K.inverse()*b.transpose()*S10*L.transpose().inverse();

    //6. find eigenvalues of V'V
    MatrixXd VpV = V.transpose() * V;
    SelfAdjointEigenSolver<MatrixXd> eigensolver(VpV);

    VectorXd eigenwerte = VectorXd::Ones(p) - eigensolver.eigenvalues();

    double product;
    product = 1.0;

    for (int el = p-1 ; el > p - q - 1; el--) {

        product *= eigenwerte(el);

    }


    double f = (1 / static_cast<double>(T-fitted_lag_order));
    double ll_res = S00.determinant() * product;
    ll_res = (-0.5) * (1 / f) * log(ll_res);

    //std::cout << "restricted ll is " << ll_res << std::endl;
    //restricted is set to FALSE by default
    if (this->restricted)
    {
        fvec(0) = ll_res;
    } else {
        fvec(0) = 2.0 * (this->ll - ll_res) - this->critical_value; //this is the value of the
        //in whose zeroes we are interested in
    }


    return 0;
}

void RRR::MakeOLSEstimates() {


    int p = this->dat.cols();
    int T = this->dat.rows();

    //step one: make variables

    MatrixXd Y_luet = this->dat.bottomRows(T - this->fitted_lag_order).transpose();
    MatrixXd X_luet = MatrixXd::Zero(p*this->fitted_lag_order, T - this->fitted_lag_order);

    for (int i = this->fitted_lag_order; i > 0; i--)
    {
        X_luet.block((this->fitted_lag_order - i)*p, 0, p, T - this->fitted_lag_order) = this->dat.block(i - 1, 0, T - this->fitted_lag_order, p).transpose();
    }

    //step two: regress those variables on the intercept.

        MatrixXd intercept = MatrixXd::Ones(1,this->T-this->fitted_lag_order);
        MatrixXd ols_coeff_y = ( (Y_luet * Y_luet.transpose()).ldlt().solve(Y_luet * intercept.transpose() ));
        MatrixXd ols_coeff_x = ( (X_luet * X_luet.transpose()).ldlt().solve(X_luet * intercept.transpose() ));

        MatrixXd y_dat = Y_luet - ols_coeff_y * intercept;
        MatrixXd x_dat = X_luet - ols_coeff_x * intercept;

        //now regress y on x...

        //time to make the long-run covariance A implied here is A'
        MatrixXd Atemp = ( (x_dat * x_dat.transpose()).ldlt().solve(x_dat * y_dat.transpose() )).transpose();
        MatrixXd res = y_dat - Atemp * x_dat;
        this->Sigma = res * res.transpose() / static_cast<double>(this->T);
        this->F = MatrixXd::Zero(p*this->fitted_lag_order,p*this->fitted_lag_order);
        this->F.topRows(p) = Atemp;
        this->F.bottomRows(p*(this->fitted_lag_order-1)) = MatrixXd::Identity(p*this->fitted_lag_order,p*this->fitted_lag_order).topRows(p*(this->fitted_lag_order-1));

}

