//
// Created by Jerome Simons on 03/03/2023.
//

#include "GridEstimator.h"
#include "RRR.h"
#include <unsupported/Eigen/NonLinearOptimization>
#include <vector>
#include <iostream>
#include <parallel/algorithm>
#include "convenience_functions.cpp"



using namespace Eigen;

GridEstimator::GridEstimator(bool autocentred,
                             bool finding_weights,
                             const int & grid_points_null,
                             const Eigen::Ref<const Eigen::MatrixXd> & allweights_in,
                             const int & M,
                             const int & T,
                             const int & num_experiments,
                             const double & stepwidth_null,
                             const double & joh_critval):
                             allweights(allweights_in),
                             T(T),
                             autocentred(autocentred),
                             M(M),
                             num_experiments(num_experiments),
                             finding_weights(finding_weights),
                             grid_points_null(grid_points_null),
                             stepwidth_null(stepwidth_null),
                             joh_critval(joh_critval){

    this->lambda_lu_vec.resize(M);

    this->ml_estimates = Eigen::MatrixXd::Zero(M,13);
    this->ml_estimate = Eigen::MatrixXd::Zero(1,13);
    this->chosen_weights = Eigen::MatrixXd::Zero(M,1);

    this->g = Eigen::ArrayXXd::Zero(1,M);
    this->f = Eigen::ArrayXXd::Zero(M,grid_points_null);


    if(this->finding_weights)  //if we are finding weights, full enchilada
        //dimensionality scheme is: generator root dimension, imposed root dimension, MC reps, and multiple nulls to test
        //M,M,num_experiments,grid_points_null
    {
        this->f_t = Eigen::Tensor<double, 4>(M,M,num_experiments,grid_points_null);
        this->g_t = Eigen::Tensor<double, 3>(M,M,num_experiments);
        //for M generator roots, a select number of MC reps, and across a grid of null hypotheses
        this->johansen_lr_stats = Eigen::Tensor<double,3>(M,num_experiments,grid_points_null);
        this->rejections = Eigen::Tensor<bool,3>(M,num_experiments,grid_points_null);
        this->rejections_johansen = Eigen::Tensor<bool,3>(M,num_experiments,grid_points_null);
    }
    else                //if we are not finding weights, then no generator root dimension is necessary
    {
        this->f_t = Eigen::Tensor<double, 4>(1, M, num_experiments,grid_points_null);
        this->g_t = Eigen::Tensor<double, 3>(1, M, num_experiments);
        this->johansen_lr_stats = Eigen::Tensor<double,3>(1,num_experiments,grid_points_null);
        this->rejections_johansen = Eigen::Tensor<bool,3>(1,num_experiments,grid_points_null);
        this->rejections = Eigen::Tensor<bool,3>(1,num_experiments,grid_points_null);

    }



    this->weighted_null_density = Eigen::ArrayXXd::Zero(1,grid_points_null);
    this->rejprobs = Eigen::Array<bool,1,Eigen::Dynamic>::Zero(1,M);

    this->marginal_interval = Eigen::MatrixXd::Zero(2,5);

    //set up grid for imposed root
    std::vector<double> delta_vec_sqrt = linspace(0.,(M-1)*.25,M);
    __gnu_sequential::transform(delta_vec_sqrt.begin(),delta_vec_sqrt.end(),this->lambda_lu_vec.begin(),[&](const double &delta_sqrt){
        return 1 - pow(delta_sqrt,2)/static_cast<double>(this->T);
    });

    this->null_hyp_grid_allocated_question_mark = false;
    this->a_lambda_grid_allocated_question_mark = false;

    //tensor.ToFlatVector<float>()Tensor



}

void GridEstimator::MakeEstimates(const Eigen::Ref<const Eigen::MatrixXd> & data,
                                  const int & lag_order,
                                  const bool & bootstrap) {


    RRR Soeren(data,lag_order);

    if(bootstrap)
    {
        Soeren.MakeOLSEstimates();
        this->F = Soeren.F;
        this->Sigma = Soeren.Sigma;
    }



    for (int j=0; j<this->lambda_lu_vec.size(); j++) {

        Soeren.restricted = true;
        Soeren.GetStart(this->lambda_lu_vec[j],true);
        this->ml_estimates(j,0) = this->lambda_lu_vec[j];
        this->ml_estimates(j,1) = Soeren.ll;
        const double d = -1.*Soeren.beta(1,0)/Soeren.beta(0,0);
        this->ml_estimates(j,2) = d; //setup corresponds to beta = [I_r -D']
        this->ml_estimates(j,3) = Soeren.alpha_perp2(1,0)/Soeren.alpha_perp(0,0);
        this->ml_estimates(j,4) = Soeren.Dvec_2(0); //actually R_lu
        //checking that the restricted ll is the same
        Eigen::VectorXd fvec(1);
        fvec(0) = 0.;
        Eigen::VectorXd qcc(1);
        qcc(0) = d;
        Soeren(qcc,fvec);
        this->ml_estimates(j,5) = fvec(0);

        //Find long-run correlation
        double lr_corr = Soeren.LongRunCR();
        this->ml_estimates(j,12) = Soeren.LongRunCR();

        //prepare the Soeren class
        Soeren.restricted = false; // computes LR - critval
        Soeren.critical_value =  this->joh_critval;

        //init solver
        Eigen::HybridNonLinearSolver<RRR> solver(Soeren);
        solver.parameters.nb_of_subdiagonals = 1;
        solver.parameters.nb_of_superdiagonals = 1;
        solver.diag.setConstant(qcc.size(), 1.);
        solver.useExternalScaling = true;

        //get confidence interval around d to the right
        qcc(0) = d - .01;
        this->ml_estimates(j,6) = solver.solveNumericalDiff(qcc);
        this->ml_estimates(j,7) = solver.fvec.blueNorm();
        this->ml_estimates(j,8) = qcc(0);
        //get right end point
        qcc(0) = d + .01;
        this->ml_estimates(j,9) = solver.solveNumericalDiff(qcc);
        this->ml_estimates(j,10) = solver.fvec.blueNorm();
        this->ml_estimates(j,11) = qcc(0);

    }


}

void GridEstimator::FindMLEstimates() {

    Eigen::Index maxrow;

    this->ml_estimates.col(1).maxCoeff(&maxrow);
    this->ml_estimate = this->ml_estimates.row(maxrow);

    //Find weights
    Eigen::Index weight_row;
    (this->allweights.col(0).array() - this->ml_estimate(0,12)).abs().matrix().minCoeff(&weight_row);
    this->chosen_weights = this->allweights.rightCols(this->lambda_lu_vec.size()).row(weight_row);

}





void GridEstimator::PrepareInterval2(const int & grid_points,
                                     const int & lag_order,
                                     const Eigen::Ref<const Eigen::MatrixXd> & data) {

    if(autocentred)
    {
        //initialise H_0's to test
        double midpoint = this->ml_estimate(0,2);
        this->null_hyp_grid = std::vector<double>(grid_points,midpoint);
        //make sure it starts at position 9 (which is one before)
        //marching up, 11 to 19 -> 10 -> 18
        for(int i = (grid_points-1)/2 + 1; i<grid_points; i++) this->null_hyp_grid[i] = this->null_hyp_grid[i-1] + this->stepwidth_null;
        //marching down, 9 -> 1 OR 8 -> 0
        for(int i = (grid_points-1)/2 - 1; i>-1; i--) this->null_hyp_grid[i] = this->null_hyp_grid[i+1] - this->stepwidth_null;

        //initialise mapping, akin to (24) in EMW.
        this->a_lambda_mapping = std::vector<double>(this->lambda_lu_vec.size(),midpoint);
        //make sure it starts at position 9 (which is one before)
        //marching up, 11 to 19 -> 10 -> 18
        for(int i = (this->lambda_lu_vec.size()-1)/2 + 1; i<this->lambda_lu_vec.size(); i++) a_lambda_mapping[i] = a_lambda_mapping[i-1] / .98;
        //marching down, 9 -> 1 OR 8 -> 0
        for(int i = (this->lambda_lu_vec.size()-1)/2 - 1; i>-1; i--) a_lambda_mapping[i] = a_lambda_mapping[i+1] * .96;

        this->null_hyp_grid_allocated_question_mark = true;
        this->a_lambda_grid_allocated_question_mark = true;

    }
    else
    {
        if(!(this->null_hyp_grid_allocated_question_mark && this->a_lambda_grid_allocated_question_mark))
        {
            std::cout << "A, lambda correspondence and null hypotheses have not been allocated." << std::endl;
            throw std::invalid_argument("Terminating program because grids are not allocated.");
        }
    }




    VectorXd fvalue = VectorXd::Zero(1);
    VectorXd qcc = VectorXd::Zero(1);

    //std::cout << "Allocate densities." << std::endl;

    RRR Soeren(data,lag_order);

    //n indexes the null hypothesis to test
    //j indexes the root that we impose in estimation

    for (int j=0; j<this->lambda_lu_vec.size(); j++) {
        Soeren.restricted = true;
        Soeren.GetStart(this->lambda_lu_vec[j],true);
        qcc(0) = this->a_lambda_mapping[j];
        Soeren(qcc, fvalue);
        this->g(0,j) = fvalue(0) + log(1 / static_cast<double>(this->lambda_lu_vec.size() - lag_order));

        for (int n = 0; n < grid_points; n++) {
            //impose H_0
            qcc(0) = this->null_hyp_grid[n];
            Soeren(qcc, fvalue);
            this->f(j,n) = fvalue(0) + log(this->chosen_weights(j));
        }
    }
}


void GridEstimator::PrepareInterval3(const int & grid_points_null,
                                     const int & lag_order,
                                     const int & mc_repetition,
                                     const int & generator_root_index,
                                     const Eigen::Ref<const Eigen::MatrixXd> & data) {

    if(autocentred)
    {
        //initialise H_0's to test
        double midpoint = this->ml_estimate(0,2);
        this->null_hyp_grid = std::vector<double>(grid_points_null,midpoint);
        //make sure it starts at position 9 (which is one before)
        //marching up, 11 to 19 -> 10 -> 18
        for(int i = (grid_points_null-1)/2 + 1; i<grid_points_null; i++) this->null_hyp_grid[i] = this->null_hyp_grid[i-1] + this->stepwidth_null;
        //marching down, 9 -> 1 OR 8 -> 0
        for(int i = (grid_points_null-1)/2 - 1; i>-1; i--) this->null_hyp_grid[i] = this->null_hyp_grid[i+1] - this->stepwidth_null;

        //initialise mapping, akin to (24) in EMW.
        this->a_lambda_mapping = std::vector<double>(this->lambda_lu_vec.size(),midpoint);
        //make sure it starts at position 9 (which is one before)
        //marching up, 11 to 19 -> 10 -> 18
        for(int i = (this->lambda_lu_vec.size()-1)/2 + 1; i<this->lambda_lu_vec.size(); i++) this->a_lambda_mapping[i] = this->a_lambda_mapping[i-1] / .98;
        //marching down, 9 -> 1 OR 8 -> 0
        for(int i = (this->lambda_lu_vec.size()-1)/2 - 1; i>-1; i--) this->a_lambda_mapping[i] = this->a_lambda_mapping[i+1] * .96;

        this->null_hyp_grid_allocated_question_mark = true;
        this->a_lambda_grid_allocated_question_mark = true;

    }
    else
    {
        if(!(this->null_hyp_grid_allocated_question_mark && this->a_lambda_grid_allocated_question_mark))
        {
            std::cout << "A, lambda correspondence and null hypotheses have not been allocated." << std::endl;
            throw std::invalid_argument("Terminating program because grids are not allocated.");
        }
    }


    VectorXd fvalue = VectorXd::Zero(1);
    VectorXd qcc = VectorXd::Zero(1);

    //std::cout << "Allocate densities." << std::endl;

    RRR Soeren(data,lag_order);

    //n indexes the null hypothesis to test
    //j indexes the root that we impose in estimation

    //LR test stats for Johansen procedure
    double ll_joh_maximised=0.;
    double ll_joh_null=0.;


    for (int j=0; j<this->lambda_lu_vec.size(); j++) {
        Soeren.restricted = true;
        Soeren.GetStart(this->lambda_lu_vec[j],true);
        if(j==0) ll_joh_maximised = Soeren.ll; //siphon off the Johansen log-likelihood
        qcc(0) = this->a_lambda_mapping[j];
        Soeren(qcc, fvalue);
        this->g_t(generator_root_index,j,mc_repetition) = fvalue(0) + log(1 / static_cast<double>(this->lambda_lu_vec.size() - lag_order));

        for (int n = 0; n < grid_points_null; n++) {
            //impose H_0
            qcc(0) = this->null_hyp_grid[n];
            Soeren(qcc, fvalue);
            if(j==0)
            {
                ll_joh_null = fvalue(0); //siphon off the Johansen log-likelihood
                //this->johansen_lr_stats = Eigen::Tensor<double,3>(1,num_experiments,grid_points_null);
                this->johansen_lr_stats(generator_root_index,mc_repetition,n) = 2 * (ll_joh_maximised-ll_joh_null);
            }
            //this->f_t = Eigen::Tensor<double,4>(M,M,grid_points_null,num_experiments);
            this->f_t(generator_root_index,j,mc_repetition,n) = fvalue(0) + log(this->chosen_weights(j));
            this->f(j,n) = fvalue(0) + log(this->chosen_weights(j));
        }
    }
}


void GridEstimator::FindInterval3(const double &critical_value) {

    int root_dim = this->M;
    if(!this->finding_weights) root_dim = 1;
    Eigen::array<int,1> imposed_dim({1}); //second dimension is imposed

    Eigen::array<int,3> shape({root_dim,1,num_experiments});
    Eigen::array<int,3> bcast2({1,M,1});
    //now a 2-tensor with dimension root_dim by num_experiments, result of integration over second (imposed) dimension
    //dimensionality scheme is: generator root dimension, imposed root dimension, MC reps, and multiple nulls to test
    //M,M,num_experiments,grid_points_null
    Eigen::Tensor<double, 2> g_bar_t = this->g_t.maximum(imposed_dim) + (this->g_t - this->g_t.maximum(imposed_dim).reshape(shape).broadcast(bcast2)).exp().sum(imposed_dim).log();

    //coming up next: integrate the null density over the imposed dimension M, M, grid_points_null, num_experiments
    Eigen::array<int,4> null_density_shape_post_reduction({root_dim,1,this->num_experiments,this->grid_points_null});
    Eigen::array<int,4> bcast_post_reduction({1,M,1,1});

    //now a 3 tensor with dimension generator_root_dim by num_experiments by grid_points_null
    Eigen::Tensor<double,3> f_bar_t = this->f_t.maximum(imposed_dim) + (this->f_t - this->f_t.maximum(imposed_dim).reshape(null_density_shape_post_reduction).broadcast(bcast_post_reduction)).exp().sum(imposed_dim).log();

    //rejection probabilities, of dimension generator_root_dim by num_experiments by grid_points_null
    //TODO recognise this is the naive estimator, no importance sampling
    //TODO stretch g along to match the dimension
    Eigen::array<int,3> shape_extender({root_dim,this->num_experiments,1});
    Eigen::array<int,3> g_stretcher({1,1,this->grid_points_null});

    //std::cout << "g_bar_t is " << g_bar_t.reshape(shape_extender).broadcast(g_stretcher) << std::endl;
    //std::cout << "f_bar_t is " << f_bar_t << std::endl;



    this->rejections = g_bar_t.reshape(shape_extender).broadcast(g_stretcher) > critical_value + f_bar_t;

    std::cout << "Rejections on null grid for EMW are\n" << this->rejections << std::endl;

    //this->weighted_null_density = this->f.colwise().maxCoeff() + (this->f.rowwise() - this->f.colwise().maxCoeff()).exp().colwise().sum().log();

    //this->rejprobs = g_bar >  critical_value + this->weighted_null_density;

    //Time for Soeren
    //this->rejections_johansen = Eigen::Tensor<bool,3>(1,num_experiments,grid_points_null);
    this->rejections_johansen = this->johansen_lr_stats > this->joh_critval;

    std::cout << "Rejections on null grid for Johansen are\n" << this->rejections_johansen << std::endl;






}



void GridEstimator::FindInterval2(const double & critical_value) {

    //find integral of g
    double g_bar =  this->g.maxCoeff() + log((this->g - this->g.maxCoeff()).exp().sum());

    this->weighted_null_density = this->f.colwise().maxCoeff() + (this->f.rowwise() - this->f.colwise().maxCoeff()).exp().colwise().sum().log();

    this->rejprobs = g_bar >  critical_value + this->weighted_null_density;

    std::cout << "rejection probabilities on basic null grid are\n" << this->rejprobs << std::endl;
    std::cout << "g_bar is " << g_bar << std::endl;
    std::cout << "f_bar is " << this->weighted_null_density << std::endl;


    //find not rejected values
    bool first = true;
    bool last2 = false;

    //row 1 EMW
    for(int n=0;n<this->null_hyp_grid.size();n++)
    {
        if(!this->rejprobs(n) && first)
        {
            this->marginal_interval(0,0) = this->null_hyp_grid[n];
            first = false;
        }
        if(this->rejprobs(n) && !first && !last2)
        {
            this->marginal_interval(0,1) = this->null_hyp_grid[n-1];
            last2 = true;
        }
    }
}

void GridEstimator::SetHypothesisGrid(const double & midpoint, const int & grid_points) {


    //initialise H_0's to test
    this->null_hyp_grid = std::vector<double>(grid_points,midpoint);
    //make sure it starts at position 9 (which is one before)
    //marching up, 11 to 19 -> 10 -> 18
    for(int i = (grid_points-1)/2 + 1; i<grid_points; i++) this->null_hyp_grid[i] = this->null_hyp_grid[i-1] + this->stepwidth_null;
    //marching down, 9 -> 1 OR 8 -> 0
    for(int i = (grid_points-1)/2 - 1; i>-1; i--) this->null_hyp_grid[i] = this->null_hyp_grid[i+1] - this->stepwidth_null;

    this->null_hyp_grid_allocated_question_mark = true;

}

void GridEstimator::SetALambdaCorrespondence(const double & midpoint, const int & gridpoints) {
    //initialise mapping, akin to (24) in EMW.
    this->a_lambda_mapping = std::vector<double>(this->lambda_lu_vec.size(),midpoint);
    //make sure it starts at position 9 (which is one before)
    //marching up, 11 to 19 -> 10 -> 18
    for(int i = (this->lambda_lu_vec.size()-1)/2 + 1; i<this->lambda_lu_vec.size(); i++) a_lambda_mapping[i] = a_lambda_mapping[i-1] / .98;
    //marching down, 9 -> 1 OR 8 -> 0
    for(int i = (this->lambda_lu_vec.size()-1)/2 - 1; i>-1; i--) a_lambda_mapping[i] = a_lambda_mapping[i+1] * .96;

    this->a_lambda_grid_allocated_question_mark = true;
}

void GridEstimator::WriteOutputGrid(const std::string &outputPath,
                                const double & target_size,
                                const int & lag_order) {
    std::string filename = "grid_estimate";
    //attach some extra information
    MatrixXd dgp_info(this->ml_estimates.rows(),3);
    for(int i = 0; i<dgp_info.rows();i++)
    {
        dgp_info(i,0) = target_size;
        dgp_info(i,1) = this->lambda_lu_vec.back();
        dgp_info(i,2) = lag_order;
    }

    MatrixXd output_grid(this->ml_estimates.rows(),this->ml_estimates.cols()+3);
    output_grid << this->ml_estimates, dgp_info;
    writeToCSVfile(outputPath+filename+".csv",output_grid);
}

void GridEstimator::WriteOutput(const double &target_size,const int & lag_order,
                                const std::string &outputPath) {

    //find not rejected values
    bool first = true;
    bool last = false;

    //row 1 EMW

    for(int n=0;n<this->null_hyp_grid.size();n++)
    {
        if(!this->rejections(0,0,n) && first)
        {
            this->marginal_interval(0,0) = this->null_hyp_grid[n];
            first = false;
        }
        if(this->rejections(0,0,n) && !first && !last)
        {
            this->marginal_interval(0,1) = this->null_hyp_grid[n-1];
            last = true;
        }
    }
    first = true;
    last = false;
    //row 2 Johansen
    for(int n=0;n<this->null_hyp_grid.size();n++)
    {
        if(!this->rejections_johansen(0,0,n) && first)
        {
            this->marginal_interval(1,0) = this->null_hyp_grid[n];
            first = false;
        }
        if(this->rejections_johansen(0,0,n) && !first && !last)
        {
            this->marginal_interval(1,1) = this->null_hyp_grid[n-1];
            last = true;
        }
    }

    //size

    this->marginal_interval(0,2) = target_size;
    this->marginal_interval(1,2) = target_size;

    //root bottom
    this->marginal_interval(0,3) = this->lambda_lu_vec.back();
    this->marginal_interval(1,3) = 1;

    //lag order
    this->marginal_interval(0,4) = lag_order;
    this->marginal_interval(1,4) = lag_order;

    const std::string filename = "marginal_interval";
    writeToCSVfile(outputPath+filename+".csv",this->marginal_interval);


}





