//
// Created by Jerome Simons on 04/03/2023.
//

#ifndef QUASICOINTEGRATION_SIMULATOR_H
#define QUASICOINTEGRATION_SIMULATOR_H

#include <vector>
#include <Eigen/Dense>
#include "EigenMultivariateNormal.h"
#include <unsupported/Eigen/CXX11/Tensor>


typedef std::mt19937 rng_type;


class Simulator {

public:

    /**
      * Constructor that sets up a simulation exercise and allocates memory.
      *
      * @param grid_points_M is the number of grid points for generator roots. For the simulation exercises, its value should be one.
      * @param imposed_grid_dim is the number of grid points for the imposed roots, it should always be equal to 19
      * @param num_experiments specifies the number of MC repetitions.
      * @param grid_points_null defines how many points the null hypothesis grid should have. Set that to one to just get size.
      * @param T defines the sample size and is needed to calculate the implied roots.
      * @param critical_value defines the starting critical value
      * @param finding_weights checks whether we wish to simulate in the service of finding weights
      * @param fitted_lag_order is the intended lag order for the VAR
      * @param step_width_null controls the size of the
      */

    Simulator(const int & grid_points_M,
              const int & imposed_grid_dim,
              const int & num_experiments,
              const int & grid_points_null,
              const int & T,
              double critical_value,
              const bool & finding_weights,
              const int & fitted_lag_order,
              const double & step_width_null,
              const double & joh_critval);


   void SetNullHypGrid(const std::vector<double> & mynullhypgrid);


    /**
     * Method that runs the simulation experiment on the given DGP using spectral data, tests a (grid of) null(s), and reports rejection probabilities.
     * This method is not necessary to
     * @param correlation is the long-run correlation in the predictive regression model. It indexes the bias when unit roots are improperly imposed.
     * @param D is the quasi-cointegrating coefficient.
     * @param T specifies the number of MC repetitions.
     */
    void Simulate(const double & correlation,const double & D,const int & T);

    /**
    * method that runs the simulation experiment on an estimated DGP for bootstrapping critical values and reports rejection probabilities.
    *
    * @param F is the companion matrix.
    * @param Sigma is the covariance matrix of the errors.
    * @param T is the sample size.
    * @param critical_value_start is the starting critical value, set to zero by default.
    * @param root_index is the index in the generator root set just in case this method is used to allocate values for weights
    */

    void RunSimulations(const Eigen::Ref<const Eigen::MatrixXd> &F,
                        const Eigen::Ref<const Eigen::MatrixXd> &Sigma,
                        const int &T,
                        const double &critical_value_start,
                        const int & root_index,
                        const bool & autocentre);

    /**
    * Method to traverse the entire parameter space for the long-run correlation. The method works by swapping out the relevant Sigma matrix to induce
     * the desired long-run correlation.
    *
    * @param grid_points sets the number of grid points on the correlation grid.
    */

    void TraverseCorrelations(const int & grid_points);

    /**
     * Method to initialize random number generator for parallel random number generation.
     */

    void InitializeRandomNG();


    /**
     * Method to read in weights
     * @param weights_path
     * @param critical_values_path
     */

    void SetWeights(const std::string & weights_path);
    void SetCriticalValue(const std::string & critical_value_path);



    void BootstrapCriticalValue(const Eigen::Ref<const Eigen::MatrixXd> &F, const Eigen::Ref<const Eigen::MatrixXd> &Sigma,
                                const int &T,const double & target_size);

    void FindInterval(const double & critical_value);

    void WriteOutput(const std::string & outputPath) const;

    void FormatOutput();

    void SetCentringOfIntervals(const Eigen::Ref<const Eigen::MatrixXd> & D_center);



    //data structures for testing
    const int num_experiments;
    const int grid_points_M;
    const int grid_points_null;
    const int imposed_grid_dim;
    const bool finding_weights;
    const double & step_width_null;
    bool bootstrap;
    const int fitted_lag_order;
    double joh_critval;
    std::vector<int> counter;
    std::vector<Eigen::EigenMultivariateNormal<double>> random_objects;
    std::vector<rng_type::result_type> seeds;

    std::vector<double> lambda_lu_vec;
    std::vector<double> generator_root_vec;
    std::vector<double> null_hyp_grid;



    Eigen::MatrixXd weight_matrix;
    Eigen::MatrixXd D_center;
    double critical_value;


    Eigen::Tensor <double,4> f_t;
    Eigen::Tensor <double,3> g_t;
    Eigen::Tensor<double,3> johansen_lr_stats;
    Eigen::Tensor<bool,3> rejections_johansen;
    Eigen::Tensor<double,2> rejection_probabilities_johansen;
    Eigen::Tensor<bool,3> rejections;
    Eigen::Tensor<double,2> rejection_probabilities;




    //data structures for output









};


#endif //QUASICOINTEGRATION_SIMULATOR_H
