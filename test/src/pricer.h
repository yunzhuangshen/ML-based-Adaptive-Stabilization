#ifndef PRICER_H
#define PRICER_H

#include <vector>
#include <math.h>
#include <cmath>
#include <math.h>       /* sqrt */
#include <numeric>      // iota
#include <time.h>
#include <sys/time.h>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <chrono>
#include "util.h"


namespace GCP{
using namespace std;
class Pricer {

public:
    int method;
    double cutoff;
    int sample_size;
    int niterations;
    int basis_factor;
    int upper_col_limit;

    vector<double> dual_values;
    double heur_best_reduced_cost;
    const double INF = 1e8;

    long nb_node;
    long nb_edge;
    vector<vector<int>> new_cols;
    vector<double> new_col_rc_values;
    vector<long> sorted_indices;
    long num_neg_rc_col=0;;
    double best_rc = 0.;
    double mean_rc = 0.;
    double stdev_rc = 0.;
    double median_rc = 0.;

    virtual ~Pricer(){};
    virtual void run(){cout << "error, in pricer!!!\n\n";};
    void compute_statistics();
    void include_new_cols(std::vector<std::vector<int>>& basic_cols, vector<int>& lb_vbasis, double param);
    void add_all(std::vector<std::vector<int>>& basic_cols);
    void add_partial(std::vector<std::vector<int>>& basic_cols, double param);
    void replace_existing(vector<vector<int>>& basic_cols, vector<int>& lb_vbasis );

};

}

#endif