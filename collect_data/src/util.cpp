#include "util.h"

namespace GCP {

    void gen_random_cols(vector<vector<bool>>& adj_matrix, int seed, int nb_col, vector<vector<int>>& cols, set<string>& col_identities) {

        int nb_node = adj_matrix.size();
        cols.resize(nb_col);
        mt19937 mt(seed);
        uniform_int_distribution<int> dist(0,RAND_MAX);
        int v, idx, num;
        vector<int> candidates(nb_node);
        int nb_candidates;

        int ctr= 0;
        for(int i = 0; i < nb_col; ++i) {
            vector<int> col;

            nb_candidates = nb_node;
            for (int j = 0; j < nb_candidates; ++j){
                candidates[j] = j;
            }
            while (nb_candidates > 0){
                if (nb_candidates == nb_node){
                    idx = i % nb_node;
                } else{
                    idx = dist(mt) % nb_candidates;
                }
                v = candidates[idx];
                col.push_back(v);
                num = 0;
                for (int j = 0; j < nb_candidates; ++j){
                    if (adj_matrix[v][candidates[j]] == 0 && j != idx){
                        candidates[num] = candidates[j];
                        num++;
                    }
                }
                nb_candidates = num;
            }

            // // check repetitive columns;
            stable_sort (col.begin(), col.end());  
            stringstream ss;
            for (auto k = 0; k < col.size(); k++){
                ss << col[k] << " ";
            }
            string identity = ss.str();  
            if (col_identities.find(identity) == col_identities.end()){
                cols[ctr++] = col;
                col_identities.insert(identity);
            }
        }
        cols.resize(ctr);
    }

}