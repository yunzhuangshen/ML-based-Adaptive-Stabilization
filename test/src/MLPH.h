#ifndef MLPH_H
#define MLPH_H

#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>
#include <set>
#include <cstring>
#include <string>
#include <iomanip>
#include "gurobi_c++.h"
#include "pricer.h"

using namespace std;

namespace GCP{
    class MLPH: public Pricer{
        std::vector<float> predicted_value;
        vector<vector<int>> mis_set;
        const vector<vector<bool>> adj_matrix;
        const vector<double> degree_norm;
        std::vector<double> objs;
        vector<double> dists;
        double best_obj=0.;
        int seed;
        public:
            double start_time;
            float min_cbm, max_cbm;
            float min_rbm, max_rbm;
            double max_dual;
            std::vector<float> ranking_scores;
            std::vector<float> corr_xy;
            std::vector<double> bound_norm;
            double b0;
            double b1;
            double c0;
            double c1; 
            double c2; 
            double c3; 
            double c4;
            double b;
            std::set<std::string> identites;
            MLPH(int _method, double b0, double b1, int _n, int _sample_size, const vector<double>& _degree_norm, 
            const vector<vector<bool>>& _adj_matrix, const vector<double>& _dual_values, int upper_col_limit);
            MLPH(int _method, double cutoff, int _n, int _sample_size, int _nb_iteration, const vector<double>& _degree_norm, 
            const vector<vector<bool>>& _adj_matrix, const vector<double>& _dual_values, int upper_col_limit, int seed);
            void run();
            void random_sampling();
            double calc_dist(vector<int> sample);
            void make_prediction(int ith_iteration);            
            void run_iteration(int ith_iteration);           
            void compute_correlation_based_measure();
            void compute_ranking_based_measure();    
            void compute_bound();
            void fast_nondominated_sort();
            bool check_dominance(int ind1, int ind2);
    };
}

#endif
