//
// Created by Jerome Simons on 03/03/2023.
//

#ifndef QUASICOINTEGRATION_GRIDESTIMATOR_H
#define QUASICOINTEGRATION_GRIDESTIMATOR_H


#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>


class GridEstimator {

public:

    /**
      * Constructor that sets up a simulation exercise and allocates memory.
      *
      * @param centred that checks whether to center the null hypotheses on the ML estimate
      * @param grid_points_null number of points on the null hypothesis grid
      * @param allweights_in matrix with weights
      * @param M number of grid points of the imposed root
      * @param T defines the sample size and is needed to calculate the implied roots.
      */

    GridEstimator(bool autocentred,
                  bool finding_weights,
                  const int & grid_points_null,
                  const Eigen::Ref<const Eigen::MatrixXd> & allweights_in,
                  const int & M,
                  const int & T,
                  const int & num_experiments,
                  const double & stepwidth_null,
                  const double & joh_critval);

    const Eigen::MatrixXd allweights;

    std::vector<double> lambda_lu_vec;
    std::vector<double> a_lambda_mapping;
    std::vector<double> null_hyp_grid;

    void SetHypothesisGrid(const double & midpoint,const int & gridpoints);
    void SetALambdaCorrespondence(const double & midpoint,const int & gridpoints);

    int T;
    void MakeEstimates(const Eigen::Ref<const Eigen::MatrixXd> & data,
                       const int& lag_order,
                       const bool & bootstrap);


    void WriteOutput(const std::string & outputpath);
    Eigen::MatrixXd ml_estimates;
    Eigen::MatrixXd ml_estimate;
    Eigen::ArrayXd chosen_weights;
    Eigen::MatrixXd F;
    Eigen::MatrixXd Sigma;
    bool autocentred;
    bool finding_weights;
    bool null_hyp_grid_allocated_question_mark;
    bool a_lambda_grid_allocated_question_mark;

    double stepwidth_null;
    const double joh_critval;







    void FindMLEstimates();
    /**
     *
     * @param grid_points refers to the size of the null hypothesis grid
     * @param lag_order is the intended lag order of the fitted model
     * @param data is the data set the f,g are allocated for
     */

    void PrepareInterval2(const int & grid_points,
                          const int & lag_order,
                          const Eigen::Ref<const Eigen::MatrixXd> & data);

    /**
     *
     * @param grid_points refers to the size of the null hypothesis grid
     * @param lag_order is the intended lag order of the fitted model
     * @param mc_repetition is the relevant mc rep
     * @param generator_root_index when finding weights, this index keeps track of which root was used to generate the data
     * @param data is the data set the f,g are allocated for
     */


    void PrepareInterval3(const int & grid_points,
                          const int & lag_order,
                          const int & mc_repetition,
                          const int & generator_root_index,
                          const Eigen::Ref<const Eigen::MatrixXd> & data);
    void FindInterval2(const double & critical_value);

    void FindInterval3(const double & critical_value);


    //make tensor cousins to f and g
    Eigen::Tensor <double,4> f_t;
    Eigen::Tensor <double,3> g_t;
    Eigen::Tensor<bool,3> rejections;

    Eigen::Tensor<double,3> johansen_lr_stats;
    Eigen::Tensor<bool,3> rejections_johansen;


    int M;
    int num_experiments;
    int grid_points_null;

    Eigen::Array<double,1,Eigen::Dynamic> g;
    Eigen::ArrayXXd f;
    Eigen::Array<double,1,Eigen::Dynamic> weighted_null_density;

    Eigen::Array<bool,1,Eigen::Dynamic> rejprobs;

    Eigen::Array<double,1,Eigen::Dynamic> marginal_interval;



};



#endif //QUASICOINTEGRATION_GRIDESTIMATOR_H
