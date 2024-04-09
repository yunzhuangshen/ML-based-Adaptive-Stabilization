#include "pricer.h"
#include <stdlib.h>     /* abs */
#include <list>
#include <random>
#include <bits/stdc++.h>
#include <limits>
#include <cassert>
#include <cmath>

#define pow2(n) ( 1 << (n) )

namespace GCP{
using namespace std;

    void Pricer::compute_statistics(){
                // calculate statistics

        num_neg_rc_col = 0;

        for (int i = 0; i < new_col_rc_values.size();i++)
            num_neg_rc_col += new_col_rc_values[i] < -eps;
        sorted_indices = sort_indexes_inc(new_col_rc_values);
        best_rc = new_col_rc_values[sorted_indices[0]];
    }

     void Pricer::include_new_cols(vector<vector<int>>& basic_cols, vector<int>& lp_basis, double param){
        add_partial(basic_cols, param);
    }


    // complexity: O(n)
    void Pricer::add_all(vector<vector<int>>& basic_cols){
        // cout << "column selection method: add_all\n";
        basic_cols.insert(basic_cols.end(), new_cols.begin(), new_cols.end());
    }

    // complexity bounded by sorting: O(n log(n))
    void Pricer::add_partial(vector<vector<int>>& basic_cols, double param){
        // cout << "column selection method: add_partial\n";

        int tmp = upper_col_limit * 1;
        for (long i = 0; i < new_cols.size() && i < tmp && new_col_rc_values[sorted_indices[i]] < -eps; ++i)
            basic_cols.push_back(new_cols[sorted_indices[i]]);
    }

    void Pricer::replace_existing(vector<vector<int>>& basic_cols, vector<int>& lp_basis){
        // cout << "column selection method: replace_existing\n";

        int nbasics = 0;
        vector<double> tot_nrcs;
        vector<vector<int>>new_basic_cols;
        vector<vector<int>>tot_cols;

        for (auto i = 0; i < lp_basis.size(); i++){
            if (lp_basis[i]==0) {
                nbasics++;
                new_basic_cols.push_back(basic_cols[i]);
            }else {
                double tmp = 1;
                for (auto v : basic_cols[i])
                    tmp -= dual_values[v];
                tot_nrcs.push_back(tmp);
                tot_cols.push_back(basic_cols[i]);
            }
        }

        tot_cols.insert(tot_cols.end(), new_cols.begin(), new_cols.end());
        tot_nrcs.insert(tot_nrcs.end(), new_col_rc_values.begin(), new_col_rc_values.end());

        sorted_indices = sort_indexes_inc(tot_nrcs);

        vector<vector<long>> selected_col_indices(nb_node,vector<long>());

        bool stop = false;
        long tot_added_cols=0;
        for(auto i = 0; i < sorted_indices.size() && !stop; i++){
            auto& col = tot_cols[sorted_indices[i]];
            for (auto node : col){
                if (selected_col_indices[node].size()<10){
                    selected_col_indices[node].push_back(sorted_indices[i]);
                    tot_added_cols++;
                    if (tot_added_cols >= 10 * nb_node) stop = true;
                    break;
                }
            }
        }

        for (auto& indices:selected_col_indices){
            for (auto col_idx : indices){
                new_basic_cols.push_back(tot_cols[col_idx]);
            }
        }

        basic_cols.clear(); 
        basic_cols.insert(basic_cols.end(), new_basic_cols.begin(), new_basic_cols.end());
    }
};


