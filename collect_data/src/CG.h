#ifndef CG_H
#define CG_H
#include "instance.h"
#include "gurobi_c++.h"
#include "util.h"

using namespace std;

namespace GCP{

    class CG{
        Instance& g;
        int method_gen_col;
        int cs_method;
        int pricing_method;
        int init_col_factor;
        int lp_method;
        int seed;
        double cutoff;
        double heur_pricer_cutoff;
        int cg_iter_cutoff;
        int thread_limit=1;

    public:
        int init_distinct_col_size; 
        double min_reduced_cost_exact;
        vector<vector<int>> adj_list;
        vector<vector<bool>> adj_matrix;
        vector<int> optimal_mis;
        vector<double> dual_value;
        vector<double> dual_estimates;

        double time_duration_master=0;
        double time_duration_pricing_exact=0;
        double time_duration_pricing_heur=0;

        bool lp_optimal=false;
        vector<int> lp_vbasis;
        vector<double> lp_val;
        int heurFails = 0;
        int num_heur_runs_success = 0;
        int degenerateCtr = 0;
        double rmp_obj = 0.0;
        double rmp_obj_first = 0.0;

        double rmp_obj_pre = 0.;
        int cg_iters = 0;
        int num_init_col = 0;
        vector<vector<int>> col_set;
        set<string> col_identities;


        CG(Instance& g, int method_gen_col, int pricing_method, int init_col_factor, double cutoff, double heur_pricer_cutoff, 
            int _thread_limit, int _seed);
        CG(Instance& g, int method_gen_col, int cs_method, int pricing_method, int init_col_factor, double cutoff, int cg_iter_cutoff, double heur_pricer_cutoff, 
            int _thread_limit, int _seed);

        CG(Instance& g0, int pricing_method, double cutoff, double heur_pricer_cutoff, 
            int _thread_limit, int _seed);

        void init_rmp();
        void select_columns(vector<vector<int>>& cand_cols);
        void check_feasibility();

        void init_params();
        bool solve_RMP_LP(int LP_method, double lp_cutoff);
        bool solve_mwis_gurobi(double cutoff, double& min_rc);
        void solve(std::ofstream* output_file_cg_stats=nullptr);        
        void record_training_data(string fpath);
    };
}

#endif
