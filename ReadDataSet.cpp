//
// Created by Jerome Simons on 03/03/2023.
//

#include "ReadDataSet.h"
#include "csv.h"
#include <string>
#include "convenience_functions.cpp"

ReadDataSet::ReadDataSet(string & data_path,string & column1,string & column2) {


    io::CSVReader<2> in(data_path);
    in.read_header(io::ignore_no_column,column1,column2);
    double p1,p2=0.;
    vector<std::tuple<double,double>> observations;

    while(in.read_row(p1,p2)){
        observations.emplace_back(p1,p2);
        //cout << "Reading in pair: " << p1 << " and " << p2 << endl;
    }

    this->dat.resize(observations.size(),2);
    int i=0;
    for(auto obs : observations) {
        this->dat(i,0) = get<0>(obs);
        this->dat(i,1) = get<1>(obs);
        i++;
    }

}

void ReadDataSet::ReadInWeightsCritVal(const string &weights_path, const string &critical_value_path) {

    io::CSVReader<1> critval_in(critical_value_path);
    critval_in.read_row(this->critical_value);
    this->weights = load_csv<MatrixXd>(weights_path);
}


ReadDataSet::ReadDataSet(const string &path, int type) {

    this->dat.resize(0,0);

    switch (type) {
        case 1: //weights
        {
            this->weights = load_csv<MatrixXd>(path);
        }
        case 2: //critical value
        {
            io::CSVReader<1> critval_in(path);
            critval_in.read_row(this->critical_value);
            this->weights.resize(0,0);
        }
    }

}

void ReadDataSet::ReadInWeights(string weightspath) {
    this->weights = load_csv<MatrixXd>(weightspath);
    this->critical_value = 0;

}


