#include <iostream>
#include "ReadDataSet.h"
#include "GridEstimator.h"
#include "Simulator.h"


using namespace std;




int main(int argc, char *argv[]) {
    const int mode = atof(argv[1]); // one of three 1: load data and produce CIs, 2: simulate data to get power and size 3: compute weights

    switch (mode) {

        case 1:
        {
            //TODO empirical things
            string data_path = argv[2];
            string column1 = argv[3];
            string column2 = argv[4];
            string weightspath = argv[5];
            string critical_value_path = argv[6];
            const int M = atof(argv[7]);
            const int lag_order = atof(argv[8]);
            const int grid_points = atof(argv[9]);
            const int step_width_null = atof(argv[10]);
            const double joh_critval = atof(argv[11]);

            //read data and weights
            ReadDataSet MyData(data_path,column1,column2);
            MyData.ReadInWeightsCritVal(weightspath,critical_value_path);

            GridEstimator MyGridEstimate(true,false,grid_points,MyData.weights,M,MyData.dat.rows(),1,step_width_null,joh_critval);
            //set a matrix with grid estimates
            MyGridEstimate.MakeEstimates(MyData.dat,lag_order, false);
            //finds weights and correlation etc. across grid
            MyGridEstimate.FindMLEstimates();
            //
            MyGridEstimate.PrepareInterval2(grid_points,lag_order,MyData.dat);
            MyGridEstimate.FindInterval2(0.);

            MyGridEstimate.PrepareInterval3(grid_points,lag_order,0,0,MyData.dat);
            MyGridEstimate.FindInterval3(0.);

            break;
        }

        case 2: {
            //the above but bootstrapping for critical values
            string data_path = argv[2];
            string column1 = argv[3];
            string column2 = argv[4];
            string weightspath = argv[5];
            const int imposed_grid_length = atof(argv[6]);
            const int lag_order = atof(argv[7]);
            const int grid_points_null = atof(argv[8]);
            const string outputpath = argv[9];
            const double target_size = atof(argv[10]);
            const double starting_cv = atof(argv[11]);
            const double step_width_null = atof(argv[12]);
            const double joh_critval = atof(argv[13]);
            const int num_experiments = atof(argv[14]);


            //read data and weights but no critical value
            ReadDataSet MyData(data_path,column1,column2);
            MyData.ReadInWeights(weightspath);

            GridEstimator MyGridEstimate(true,false,grid_points_null,MyData.weights,imposed_grid_length,MyData.dat.rows(),1,step_width_null,joh_critval);
            //set a matrix with grid estimates, bootstrap option set to true
            MyGridEstimate.MakeEstimates(MyData.dat,lag_order,true);
            //finds weights and correlation etc. across grid
            MyGridEstimate.FindMLEstimates();

            //TODO check all input parameters and move them around
            Simulator BootStrapCriticalValue(1,imposed_grid_length,num_experiments,grid_points_null,MyData.dat.rows(),starting_cv,false,lag_order,step_width_null,joh_critval);
            MyGridEstimate.SetHypothesisGrid(MyGridEstimate.ml_estimate(0,2),grid_points_null);
            BootStrapCriticalValue.SetNullHypGrid(MyGridEstimate.null_hyp_grid);
            BootStrapCriticalValue.InitializeRandomNG();
            BootStrapCriticalValue.weight_matrix = MyGridEstimate.allweights;
            BootStrapCriticalValue.SetCentringOfIntervals(MyGridEstimate.ml_estimate.block(0,2,1,1));
            //std::cout << "F coming out" << MyGridEstimate.F << std::endl;
            //std::cout << "Sigma " << MyGridEstimate.Sigma << std::endl;
            BootStrapCriticalValue.BootstrapCriticalValue(MyGridEstimate.F,MyGridEstimate.Sigma,MyData.dat.rows(),target_size);

            std::cout << "Critical value is " << BootStrapCriticalValue.critical_value << std::endl;

            MyGridEstimate.PrepareInterval3(grid_points_null,lag_order,0,0,MyData.dat);
            MyGridEstimate.FindInterval3(BootStrapCriticalValue.critical_value);

            //write output for empirical power
            BootStrapCriticalValue.FormatOutput();
            BootStrapCriticalValue.WriteOutput(outputpath);
            MyGridEstimate.WriteOutput(outputpath);






            break;
        }
        case 3: {
            //furnish power study
            break;
        }
        default:
            std::cout << "Please select a mode 1,2,3; exiting programme." << std::endl;
    }

    return 0;
}
