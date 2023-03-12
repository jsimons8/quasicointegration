#include "GenerateDataSet.h"

using namespace Eigen;


GenerateDataSet::GenerateDataSet(int p,int T, int k_dgp,const Eigen::Ref<const Eigen::MatrixXd> & F):p(p),T(T),k_dgp(k_dgp),F(F){}

Eigen::MatrixXd GenerateDataSet::Generate(const Eigen::Ref<const Eigen::MatrixXd> &error_mat) {

    //this method sets the internal variables
    //initialise the VAR at zero, this would be the line to change if it were ever something else
    VectorXd y_init = VectorXd::Zero(k_dgp*p);
    VectorXd y_init_r = y_init;
    Eigen::VectorXd y = VectorXd::Zero(T*p);
    //reverse the order of y_init, currently it's meaningless because it's all zeroes anyway
    for (int j = k_dgp; j > 0; j--) {//segment(i,n)
        y_init_r.segment((k_dgp-j)*p,p) = y_init.segment((j-1)*p,  p);
    }
    //make error term

    //retrieve T samples
    //error_mat = error_mat_global.block(sim_iter*T, 0, T, p);
    //error_mat_aug << error_mat, error_mat_add;
    //change the storage order into RowMajor and then map to a vector/vectorise
    Matrix<double, Dynamic, Dynamic, RowMajor> error_mat_row(error_mat);
    Map<VectorXd> error_vec(error_mat_row.data(), p*T);
    //int fs = F.size();
    //cout << "the error_vec\n" << error_vec << endl;
    y.segment((T - k_dgp)*p, k_dgp*p) = y_init_r;
    //implement lin. recurrence relation
    /*
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> ColumnVector;
    ColumnVector column = row.transpose(); */
    MatrixXd A = F.topRows(p);
    //A << MatrixXd::Identity(p, p), MatrixXd::Identity(p, p), MatrixXd::Identity(p, p);
    for (int s = T-k_dgp-1; s >= 0; s--) {
        //y.segment((s - 1)*p, k_dgp*p) = ((F * y.segment(s*p, k_dgp*p)) + error_vec.segment((s-1)*p, k_dgp*p)).eval();
        y.segment(s*p, p) = (A * y.segment((s + 1)*p, k_dgp*p) + error_vec.segment(s*p, p)).eval();//) ).eval();
    }
    //reverse order again///

    Eigen::MatrixXd dat(T,p);

    for (int t = 0; t < T; t++)
    {
        dat.block(t, 0, 1, p) = y.segment((T - 1 - t)*p, p).transpose();
    }

    return dat;
}


