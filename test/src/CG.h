#ifndef CG_H
#define CG_H
#include "instance.h"
#include "gurobi_c++.h"
#include "util.h"

using namespace std;

namespace GCP{

    class CG{
        Instance& g;
        int pricing_method;
        int lp_method;
        int seed;
        double cutoff;
        int thread_limit=1;
        double epsilon_coef;
        double epsilon;
    public:
        double min_reduced_cost;
        vector<vector<int>> adj_list;
        vector<vector<bool>> adj_matrix;
        vector<int> best_mis;
        vector<double> dual_value;
        vector<double> dual_estimates;
        vector<double> lps;

        bool lp_optimal=false;
        double rmp_obj = 0.0;
        double rmp_obj_first = 0.0;
        double rmp_obj_pre = 0.;

        double gap = 0;
        double lagrangian_bound = 0;

        int cg_iters = 0;
        double lptime=0;
        double pricingtime=0;
        vector<vector<int>> col_set;
        set<string> col_identities;

        CG(Instance& g, double epsilon, int seed, double cutoff, int _thread_limit);
        bool rmp_dual(int method, double lp_cutoff);
        bool solve_mwis_gurobi(double cutoff, double& min_rc);
        bool solve_mwis_tsm(double cutoff, double& min_rc);
        bool solve_mwis_lscc(double cutoff, double& min_rc);
        void save_dual(std::string path);
        void solve(bool use_heuristic);        
    };
}

#endif
