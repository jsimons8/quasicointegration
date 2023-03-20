//
// Created by Jerome Simons on 04/03/2023.
//

#include "Simulator.h"
#include "GridEstimator.h"
#include "MakeParams.h"
#include <Eigen/Dense>
#include <parallel/algorithm>
#include "convenience_functions.cpp"
#include "GenerateDataSet.h"
#include "ReadDataSet.h"
#include <string>     // std::string, std::to_string
#include <iostream>   // std::cout


using namespace Eigen;
Simulator::Simulator(const int & grid_points_M,
                     const int & imposed_grid_dim,
                     const int & num_experiments,
                     const int & grid_points_null,
                     const int & T,
                     double critical_value,
                     const bool & finding_weights,
                     const int & fitted_lag_order,
                     const double & step_width_null,
                     const double & joh_critval):
                     num_experiments(num_experiments),
                     grid_points_M(grid_points_M),
                     grid_points_null(grid_points_null),
                     critical_value(critical_value),
                     finding_weights(finding_weights),
                     imposed_grid_dim(imposed_grid_dim),
                     fitted_lag_order(fitted_lag_order),
                     step_width_null(step_width_null),
                     joh_critval(joh_critval){

    //set up generator roots
    this->lambda_lu_vec.resize(grid_points_M);

    if(grid_points_M>1)
    {
        //set up grid, note that in the power study/generic simulation exercise this will be a singleton
        std::vector<double> delta_vec_sqrt = linspace(0.,.25*(grid_points_M-1),grid_points_M);
        __gnu_sequential::transform(delta_vec_sqrt.begin(),delta_vec_sqrt.end(),this->lambda_lu_vec.begin(),[&](const double &delta_sqrt){
            return 1 - pow(delta_sqrt,2)/static_cast<double>(T);
        });

    }




    //set up counting vector
    this->counter.resize(num_experiments);
    iota(this->counter.begin(),this->counter.end(),0);

    if(this->finding_weights)  //if we are finding weights, full enchilada
        //dimensionality scheme is: generator root dimension, imposed root dimension, MC reps, and multiple nulls to test
        //M,M,num_experiments,grid_points_null
    {
        this->f_t = Eigen::Tensor<double, 4>(this->grid_points_M,this->imposed_grid_dim,num_experiments,grid_points_null);
        this->g_t = Eigen::Tensor<double, 3>(this->grid_points_M,this->imposed_grid_dim,num_experiments);
        this->johansen_lr_stats = Eigen::Tensor<double,3>(this->grid_points_M,num_experiments,grid_points_null);
        this->rejections = Eigen::Tensor<bool,3>(this->grid_points_M,num_experiments,grid_points_null);
        this->rejections_johansen = Eigen::Tensor<bool,3>(this->grid_points_M,num_experiments,grid_points_null);
        this->rejection_probabilities = Eigen::Tensor<double, 2>(this->grid_points_M,this->grid_points_null);
        this->rejection_probabilities_johansen = Eigen::Tensor<double, 2>(this->grid_points_M,this->grid_points_null);
    }
    else                //if we are not finding weights, then no generator root dimension is necessary
    {
        this->f_t = Eigen::Tensor<double, 4>(1, this->imposed_grid_dim, num_experiments,grid_points_null);
        this->g_t = Eigen::Tensor<double, 3>(1, this->imposed_grid_dim, num_experiments);
        this->johansen_lr_stats = Eigen::Tensor<double,3>(1,num_experiments,grid_points_null);
        this->rejections = Eigen::Tensor<bool,3>(1,num_experiments,grid_points_null);
        this->rejections_johansen = Eigen::Tensor<bool,3>(1,num_experiments,grid_points_null);
        this->rejection_probabilities = Eigen::Tensor<double,2>(1,this->grid_points_null);
        this->rejection_probabilities_johansen = Eigen::Tensor<double, 2>(1,this->grid_points_null);
    }





}

void Simulator::Simulate(const double & correlation,const double & D,const int & T) {

    //careful here, lambda lu vec
    int root_index = 0;
    __gnu_sequential::for_each(this->lambda_lu_vec.begin(),this->lambda_lu_vec.end(),[&](double & rho){

    const int k=1;
    const int p=2;
    MatrixXd Lambda(k*p,k*p);
    MatrixXd R(k*p,k*p);
    MatrixXd Omega_predictive(p,p);
    /*
     * Set up DGP, Omega_predictive is not the cov. mat. we need, transform that
    */
    Omega_predictive << 1, correlation,
                        correlation, 1;
    Lambda <<           rho,0,
                        0,0.3;
    R <<                1, 0,
                        D, 1;
    //initialise Parameter object, find F and Sigma
    MakeParams MyParams(R, Lambda);
    MatrixXd F = MyParams.GenerateF(Lambda(0, 0));
    /*
     * Sort Sigma
     */
    MatrixXd Sigma = MyParams.PredictiveOmegaToVARSigma(Omega_predictive, false);
    const double sigma12 = Sigma(1, 0) / sqrt(Sigma(0, 0) * Sigma(1, 1));

    Sigma(0, 0) = 1.;
    Sigma(1, 1) = 1.;
    Sigma(0, 1) = sigma12;
    Sigma(1, 0) = sigma12;

    std::cout << "Omega_predictive\n" << Omega_predictive << std::endl;
    std::cout << "Corresponding Sigma\n" << Sigma << std::endl;

    /*
     * For the particular Sigma cov mat, we generate random objects, one for each thread.
     */

    /*
     * With Sigma and F in hand we can simulate
     */

    //this->RunSimulations(F,Sigma,T,0);
    this->RunSimulations(F,Sigma,T,this->critical_value,0,false);

    /*
     * Move over to the next generator root
     */


    root_index++;
    }); //end loop through generator roots, singleton in the simple case




}

void Simulator::TraverseCorrelations(const int & grid_points = 10) {

    const int k = 1;
    const int p = 2;
    std::vector<double> correlation = linspace(-.9,.9,grid_points);



    int index = 0;

    __gnu_sequential::for_each(correlation.begin(),correlation.end(),[&](double corr_predictive) {

        std::string num = std::to_string(index);


    });



}

void Simulator::InitializeRandomNG() {


    std::uniform_int_distribution<rng_type::result_type> udist;

    rng_type rng;

    // seed rng first, and store the result in a log file:
    rng_type::result_type const root_seed = 5489u;
    rng.seed(root_seed);

    this->seeds.resize(omp_get_max_threads());
    this->random_objects.resize(omp_get_max_threads());

    std::cout << "Number of threads " << omp_get_max_threads() << " reporting." << std::endl;
    for (auto & n : this->seeds) { n = udist(rng); }



}

void Simulator::RunSimulations(const Ref<const Eigen::MatrixXd> &F,
                               const Ref<const Eigen::MatrixXd> &Sigma,
                               const int &T,
                               const double &critical_value,
                               const int & root_index,
                               const bool & autocenter) {

    const int p = Sigma.rows();
    const int k = F.rows()/p;

    /*
     * Make error matrix
     */

    __gnu_sequential::transform(this->seeds.begin(),this->seeds.end(),this->random_objects.begin(),[&](rng_type::result_type seed){
        Eigen::EigenMultivariateNormal<double> errors(Eigen::VectorXd::Zero(p),Sigma,false,seed);
        return errors;
    });


    __gnu_parallel::for_each(this->counter.begin(),this->counter.end(),[&](int l) {

        GenerateDataSet MyDataSet(p, T, k, F);
        Eigen::MatrixXd error_mat = Eigen::MatrixXd::Zero(T, p);
        error_mat << this->random_objects[omp_get_thread_num()].samples(T).transpose();
        //generate a dataset
        MatrixXd dataset = MyDataSet.Generate(error_mat);

        GridEstimator MyGrid(false,false,this->grid_points_null,this->weight_matrix,this->imposed_grid_dim,T,1,this->step_width_null,this->joh_critval);

        MyGrid.SetALambdaCorrespondence(this->D_center(0,0),this->imposed_grid_dim);
        MyGrid.SetHypothesisGrid(this->D_center(0,0),this->grid_points_null);
        //cout << "centring at " << this->D_center(0,0) << endl;
        //becomes thread-specific here
        MyGrid.MakeEstimates(dataset,this->fitted_lag_order,this->bootstrap);
        MyGrid.FindMLEstimates();
        //this function allocates g and f
        MyGrid.PrepareInterval3(this->grid_points_null,this->fitted_lag_order,0,root_index,dataset);


        Eigen::array<int,4> f_input_offsets({0,0,l,0});
        Eigen::array<int,4> f_input_extents({this->grid_points_M,this->imposed_grid_dim,1,this->grid_points_null});

        /*
         *cout << "Row three of weights is " << MyGrid.allweights.row(2) << endl;
        cout << "null hyp grid at pos 12 is " << MyGrid.null_hyp_grid[12] << endl;
        cout << "a lambda corr is at pos 12 is " << MyGrid.a_lambda_mapping[12] << endl;

         */

        //cout << "centering is at " << this->D_center << endl;





        this->f_t.slice(f_input_offsets,f_input_extents) = MyGrid.f_t;

        //cout << "MyGrid.f_t " << MyGrid.f_t << endl;
        //cout << "my grid f_t " << this->f_t.slice(f_input_offsets,f_input_extents) << endl;

        //cout << "Using the chip operation for f\n" << this->f_t.chip(l,2) << endl;

        Eigen::array<int,3> g_input_offsets({0,0,l});
        Eigen::array<int,3> g_input_extents({this->grid_points_M,this->imposed_grid_dim,1});

        this->g_t.slice(g_input_offsets,g_input_extents) = MyGrid.g_t;

        /*
         * Set up transfer of Johansen quantities
         */
        Eigen::array<int,3> lr_joh_offsets({0,l,0});
        Eigen::array<int,3> lr_joh_extents({this->grid_points_M,1,this->grid_points_null});
        this->johansen_lr_stats.slice(lr_joh_offsets,lr_joh_extents) = MyGrid.johansen_lr_stats;



    }); //end loop through MC reps.



}

void Simulator::SetCriticalValue(const string & critical_value_path){
    ReadDataSet JustCriticalValue(critical_value_path,2);
    this->critical_value = JustCriticalValue.critical_value;
}

void Simulator::SetWeights(const string & weights_path) {

    ReadDataSet JustWeights(weights_path,1);
    JustWeights.ReadInWeights(weights_path);
    this->weight_matrix = JustWeights.weights;
}

void Simulator::BootstrapCriticalValue(const Ref<const Eigen::MatrixXd> &F,
                                       const Ref<const Eigen::MatrixXd> &Sigma,
                                       const int &T,
                                       const double & target_size) {

    /*
     *The idea is to use estimated parameters to bootstrap the critical value
     *The plan: first, set up a simulation, then vary the critical value until the desired size is achieved.
     */
    this->bootstrap = true;
    //cout << "F is " << F << endl;
    //cout << "Sigma is " << Sigma << endl;
    this->RunSimulations(F,Sigma,T,this->critical_value,0,false);
    //initialise
    this->FindInterval(this->critical_value);
    const int center_position = this->grid_points_null/2;
    //1,this->grid_points_null
    Eigen::array<int,1> null_grid_dim({1});
    cout << "Starting at critical value: " << this->critical_value << endl;
    double size = this->rejection_probabilities(0,center_position);
    Eigen::Tensor<double,1> rejection_probabilities(this->grid_points_M);
    rejection_probabilities = this->rejection_probabilities.minimum(null_grid_dim);
    double size_m = rejection_probabilities(0);
    while(abs(size_m - target_size) > .005)
    {
        rejection_probabilities = this->rejection_probabilities.minimum(null_grid_dim);
        size_m = rejection_probabilities(0);
        cout << "Size is " << size_m << endl;
        this->critical_value += (size_m - target_size);
        cout << "Now trying critical value: " << this->critical_value << endl;
        this->FindInterval(this->critical_value);
        cout << "Rejection probabilities are " << this->rejection_probabilities << endl;
    }

}

void Simulator::FindInterval(const double & critical_value_to_try) {

    //this function should just take g and f and find intervals



        int root_dim = this->grid_points_M;
        if(!this->finding_weights) root_dim = 1;
        Eigen::array<int,1> imposed_dim({1}); //second dimension is imposed

        Eigen::array<int,3> shape({this->grid_points_M,1,this->num_experiments});
        Eigen::array<int,3> bcast2({1,this->imposed_grid_dim,1});

        //cout << "dimension of g_t is " << this->g_t.dimensions();
        //cout << "g_t " << this->g_t << endl;
        //now a 2-tensor with dimension root_dim by num_experiments, result of integration over second (imposed) dimension
        //dimensionality scheme is: generator root dimension, imposed root dimension, MC reps, and multiple nulls to test
        //M,M,num_experiments,grid_points_null
        Eigen::Tensor<double, 2> g_bar_t = this->g_t.maximum(imposed_dim) + (this->g_t - this->g_t.maximum(imposed_dim).reshape(shape).broadcast(bcast2)).exp().sum(imposed_dim).log();

        //cout << "g_bar_t is " << g_bar_t << endl;
        //coming up next: integrate the null density over the imposed dimension M, M, grid_points_null, num_experiments
        Eigen::array<int,4> null_density_shape_post_reduction({this->grid_points_M,1,this->num_experiments,this->grid_points_null});
        Eigen::array<int,4> bcast_post_reduction({1,this->imposed_grid_dim,1,1});

        //now a 3 tensor with dimension generator_root_dim by num_experiments by grid_points_null
        Eigen::Tensor<double,3> f_bar_t = this->f_t.maximum(imposed_dim) + (this->f_t - this->f_t.maximum(imposed_dim).reshape(null_density_shape_post_reduction).broadcast(bcast_post_reduction)).exp().sum(imposed_dim).log();
        //cout << "f_bar_t has dimensions " << f_bar_t.dimensions() << endl;
        //cout << "f_bar_t is " << f_bar_t << endl;
        //rejection probabilities, of dimension generator_root_dim by num_experiments by grid_points_null
        //TODO recognise this is the naive estimator, no importance sampling
        Eigen::array<int,3> shape_extender({this->grid_points_M,this->num_experiments,1});
        Eigen::array<int,3> g_stretcher({1,1,this->grid_points_null});

        //std::cout << "g_bar_t is " << g_bar_t.reshape(shape_extender).broadcast(g_stretcher) << std::endl;
        //std::cout << "f_bar_t is " << f_bar_t << std::endl;

        Eigen::Tensor<double,3> g_bar_t_reshaped = g_bar_t.reshape(shape_extender).broadcast(g_stretcher);

        //cout << "Dimension of g_bar_t_reshaped are " << g_bar_t_reshaped.dimensions() << endl;

        this->rejections = g_bar_t.reshape(shape_extender).broadcast(g_stretcher) > critical_value_to_try + f_bar_t;

        //std::cout << "rejection probabilities on null grid are\n" << this->rejections << std::endl;

        Eigen::array<int,1> num_experiment_dim({1});
        this->rejection_probabilities = this->rejections.cast<double>().mean(num_experiment_dim);

        //this->weighted_null_density = this->f.colwise().maxCoeff() + (this->f.rowwise() - this->f.colwise().maxCoeff()).exp().colwise().sum().log();

        //this->rejprobs = g_bar >  critical_value + this->weighted_null_density;

        this->rejections_johansen = this->johansen_lr_stats > this->joh_critval;
        this->rejection_probabilities_johansen = this->rejections_johansen.cast<double>().mean(num_experiment_dim);




}

void Simulator::WriteOutput(const string &outputPath) const {

    Eigen::MatrixXd mymatrix(this->grid_points_null,3);

    for(int i=0;i<this->grid_points_M;i++)
    {
        std::string num = to_string(i);
        for (int j=0;j<this->grid_points_null;j++)
        {
            mymatrix(j,0) = this->null_hyp_grid[j];
            mymatrix(j,1) = this->rejection_probabilities(i,j);
            mymatrix(j,2) = this->rejection_probabilities_johansen(i,j);
        }

        string filename = "empirical_power_";
        writeToCSVfile(outputPath+filename+num+".csv",mymatrix);
    }




}

void Simulator::FormatOutput() {


}

void Simulator::SetNullHypGrid(const std::vector<double> & mygrid) {

    //setting up the null hypothesis grid
    this->null_hyp_grid = mygrid;

}

void Simulator::SetCentringOfIntervals(const Ref<const Eigen::MatrixXd> &D_center) {

    this->D_center = D_center;

}

