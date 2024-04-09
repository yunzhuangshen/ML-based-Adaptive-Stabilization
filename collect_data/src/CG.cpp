#include "CG.h"
#include <iostream>
#include <cassert>
#include <iostream>

namespace GCP {
using namespace std;

    CG::CG(Instance& g, int method_gen_col, int pricing_method, int init_col_factor, double cutoff, double heur_pricer_cutoff, 
            int _thread_limit, int _seed) : 
            g{g}, method_gen_col{method_gen_col}, cs_method{0}, pricing_method{pricing_method}, init_col_factor{init_col_factor}, 
            cutoff{cutoff}, cg_iter_cutoff{100000000}, heur_pricer_cutoff{heur_pricer_cutoff}, seed{_seed}, thread_limit{_thread_limit} {
        // omp_set_num_threads(thread_limit);
        lp_method = -1;
        rmp_obj = 1e8;
        rmp_obj_pre = 1e8;
        init_params();
    }

    CG::CG(Instance& g, int method_gen_col, int cs_method, int pricing_method, int init_col_factor, double cutoff, int cg_iter_cutoff, 
            double heur_pricer_cutoff, int _thread_limit, int _seed): 
            g{g}, method_gen_col{method_gen_col}, cs_method{cs_method}, pricing_method{pricing_method}, init_col_factor{init_col_factor}, 
            cutoff{cutoff}, cg_iter_cutoff{cg_iter_cutoff}, heur_pricer_cutoff{heur_pricer_cutoff}, seed{_seed}, thread_limit{_thread_limit} {
        // omp_set_num_threads(thread_limit);
        lp_method = -1;
        rmp_obj = 1e8;
        rmp_obj_pre = 1e8;
        init_params();
    }


    CG::CG(Instance& g, int pricing_method, double cutoff, double heur_pricer_cutoff, 
            int _thread_limit, int _seed) : 
            g{g}, method_gen_col{0}, cs_method{0}, pricing_method{pricing_method}, init_col_factor{10}, 
            cutoff{cutoff}, cg_iter_cutoff{100000000}, heur_pricer_cutoff{heur_pricer_cutoff}, seed{_seed}, thread_limit{_thread_limit} {
        // omp_set_num_threads(thread_limit);
        lp_method = -1;
        rmp_obj = 1e8;
        rmp_obj_pre = 1e8;
        init_params();
    }


    void CG::init_params(){

        adj_list = g.adj_list;
        adj_matrix = g.adj_matrix;
        dual_value.resize(g.size());
        fill(dual_value.begin(),dual_value.end(),0);
    }

    void CG::check_feasibility(){

        // check whether the RMP is infeasible

        vector<bool> nodeCoveredList(g.size(), false);
        for (int i = 0; i < col_set.size() ; i++){
            for (auto v : col_set[i])
                nodeCoveredList[v] = true;
        }
        
        int nb_uncovered = 0;
        for (bool val : nodeCoveredList)
            if (!val) nb_uncovered++; 

        if (nb_uncovered > 0){
            if (output_level == 1) {
                cout << "INFO: initial RMP is not feasible." << endl;
                cout << "INFO: " << nb_uncovered << "/" << g.size() << " nodes are not covered!\n" << endl; 
                cout << "INFO: " << "nodeIdx/dual values of these nodes: \n";
                for (int i = 0; i < g.size(); i++){
                    if (!nodeCoveredList[i]) {
                        cout << "INFO: "  << i << " " << dual_estimates[i] << "\n";
                    }
                }
            }
        }
        assert(nb_uncovered == 0);
    }
    void CG::select_columns(vector<vector<int>>& cand_cols){

        int nb_cand_col = cand_cols.size();
        vector<double> scores(nb_cand_col, 1);
        for (int i = 0; i < nb_cand_col; i++){
            vector<int>& cur_col = cand_cols[i];
            for (auto v : cur_col) 
                scores[i] -= dual_estimates[v];
        }

        vector<long> sorted_indices = sort_indexes_inc(scores);

        
        int end_idx = 0;
        for (; end_idx < num_init_col && end_idx < cand_cols.size(); end_idx++){
            col_set.push_back(cand_cols[sorted_indices[end_idx]]);
        }

        vector<bool> nodeCoveredList(g.size(), false);
        for (int i = 0; i < end_idx ; i++){
            for (auto v : col_set[i])
                nodeCoveredList[v] = true;
        }
        
        int nb_new_added_for_feasibility = 0;
        // select and replace from the rest random columns;
        for (int i = end_idx; i < sorted_indices.size(); i++){
            vector<int>& col = cand_cols[sorted_indices[i]];
            bool add = false;
            for (auto v : col){
                if (!nodeCoveredList[v]){
                    nodeCoveredList[v]=true;
                    add = true;
                }
            }
            
            if (add) {
                nb_new_added_for_feasibility++;
                col_set.push_back(col);
            }

            bool all_covered = true;
            for (int j = 0; j < g.size(); j++){
                if (!nodeCoveredList[j]){
                    all_covered = false;
                    break;
                }
            }
            if (all_covered) break;
        }
    }

    //TODO
    void CG::init_rmp() {

        vector<vector<int>> cand_cols;
        dual_estimates = g.estimateDual;
        num_init_col = init_col_factor * g.size();
        int cand_col_factor = 10; // factor for generating candidate columns
        int nb_cand_to_generate = g.size() * cand_col_factor;

        gen_random_cols(adj_matrix, seed, nb_cand_to_generate, cand_cols, col_identities);

        init_distinct_col_size = cand_cols.size();
        if (output_level == 1)
            cout << "INFO: nb distinct columns actually generated: " << init_distinct_col_size << endl;
        if (init_distinct_col_size > num_init_col){
            select_columns(cand_cols);
        }else{
            col_set = cand_cols;
        }

        // check whether the RMP is infeasible
        check_feasibility();
    }



    void CG::solve(std::ofstream* output_file_cg_stats){

        init_rmp();

        double start_time = get_wall_time();
        lp_optimal = false;
        min_reduced_cost_exact = -1.0;
        bool mwis_optimal = true;;
        cg_iters = 0;
        double t0;
        double rmp_duration;
        double heur_pricing_duration;
        double exact_pricing_duration;
    
        while(true){
            rmp_duration = 0;
            heur_pricing_duration = 0;
            exact_pricing_duration = 0;
            t0 = get_wall_time();
            cg_iters++; 
            
            // solve RMP
            if (output_level == 1){
                cout << "\nINFO: iteration " << cg_iters << " : " << endl;
            }
            solve_RMP_LP(lp_method, 1e8);
            if (cg_iters == 1) rmp_obj_first = rmp_obj;
            
            rmp_duration = get_wall_time()-t0;
            time_duration_master += rmp_duration;
            if (output_level == 1){
                cout << "INFO: optimal LP objective value for current RMP: " << rmp_obj << "\n";
                cout << "INFO: time used: " << rmp_duration << "\n";
            }
            
            if (output_file_cg_stats!=nullptr){
                (*output_file_cg_stats) << cg_iters << "," 
                    << get_wall_time() - start_time << ","
                    << rmp_obj << ",";
            }
            
            t0 = get_wall_time();
            auto mwis_cutoff = cutoff - (t0 - start_time);
            mwis_optimal = solve_mwis_gurobi(mwis_cutoff, min_reduced_cost_exact);

            if (mwis_optimal){
                // add new columns
                if (min_reduced_cost_exact < -0.000001){
                    col_set.push_back(optimal_mis);
                }

                // compute lagrangian bound
                if (output_file_cg_stats!=nullptr){
                    double lbound = rmp_obj / (1-min_reduced_cost_exact);
                    (*output_file_cg_stats) << lbound << "\n";
                }
            }else{
                cout << "Warning: reaching cutoff when executing exact solver!\n";
            }
            exact_pricing_duration = get_wall_time()-t0;
            time_duration_pricing_exact += exact_pricing_duration;
            
            if (!mwis_optimal) break;
            if (min_reduced_cost_exact > -0.000001) break;
            if (get_wall_time()-start_time > cutoff) break;
            if (cg_iters >= cg_iter_cutoff) break;
            t0 = get_wall_time();
        }
        
        lp_optimal = mwis_optimal && min_reduced_cost_exact > -0.000001;

    }












/**
     * LP_method: 
     *  0 - primal simplex, 
     *  1 - dual simplex
     *  2 - barrier
     *  3 - concurrent
    */
    bool CG::solve_RMP_LP(int LP_method, double lp_cutoff) {
        int nb_col = col_set.size();
        vector<vector<bool>> col_set_binary(nb_col, vector<bool>(g.size(), 0));
        long v;
        for (long i = 0; i < nb_col; ++i){
            for (long j = 0; j < col_set[i].size(); ++j){
                v = col_set[i][j];
                col_set_binary[i][v] = 1;
            }
        }
        // setup the model now
        try{
            GRBEnv *env;
            vector<GRBVar> x;
            env = new GRBEnv();
            GRBModel model = GRBModel(*env);
            model.set(GRB_IntParam_OutputFlag, 0);
            model.set(GRB_IntParam_Threads, thread_limit);
            model.set(GRB_StringAttr_ModelName, "RMP_GCP");
            model.set(GRB_IntParam_Method, LP_method);
            model.set(GRB_DoubleParam_TimeLimit, lp_cutoff);

            // Create variables and set them to be binary
            x.resize(nb_col);
            for (long i = 0; i < nb_col; ++i){
                x[i] = model.addVar(0,1,0,GRB_CONTINUOUS);
            }

            // each vertex is covered by at least one set
            vector<GRBConstr> y;
            y.resize(g.size());
            for (long j = 0; j < g.size(); ++j){
                GRBLinExpr rtot = 0;
                for (long i = 0; i < nb_col; ++i){
                    rtot += col_set_binary[i][j] * x[i];
                }
                y[j] = model.addConstr(rtot >= 1, "");
            }
            model.update();
            
            // the objective
            GRBLinExpr tot=0;
            for(long i = 0; i < nb_col; ++i){
                tot += x[i];
            }
            model.setObjective(tot,GRB_MINIMIZE);
            model.update();
            // cout << "LP solver: " << model.get(GRB_IntParam_Method) << "\n";
            model.optimize();

            // 2 - OPTIMAL
            // 3 - INFEASIBLE
            // cout << "INFO: RMP solving status: " << model.get(GRB_IntAttr_Status) << "\n";
            assert(model.get(GRB_IntAttr_Status)==GRB_OPTIMAL);

            rmp_obj_pre = rmp_obj;
            rmp_obj = model.get(GRB_DoubleAttr_ObjVal);    

            assert(!isGreater(rmp_obj, rmp_obj_pre));

            // get RMP solving statistics

            // get optimal dual
            for (long j = 0; j < g.size(); ++j){
                dual_value[j] = y[j].get(GRB_DoubleAttr_Pi);
            }


            // get optimal primal
            lp_val.resize(nb_col);
            lp_vbasis.resize(nb_col);
            fill(lp_vbasis.begin(), lp_vbasis.end(), -1);

            degenerateCtr = 0;
            int nbBasis = 0;
            int nonZeroCtr = 0;
            for (long i = 0; i < nb_col; ++i){
                auto basisStatus = x[i].get(GRB_IntAttr_VBasis);
                auto lpValue = x[i].get(GRB_DoubleAttr_X);
                lp_vbasis[i] = basisStatus;
                lp_val[i] = lpValue;
            
                if (basisStatus == 0) {
                    nbBasis++;
                    if (isEqual(lpValue, 0)) degenerateCtr++;
                    if (isGreater(lpValue, 0)) nonZeroCtr++;
                }          
            }
            
            if (output_level == 1){
                cout << "INFO: RMP solving statistics: \n";
                cout << "INFO: nb columns: " << nb_col << "; nb constrains: " << g.size()<< "\n";
                if (degenerateCtr>0) cout << "INFO: degenerate RMP; nb nonzero columns: " << nonZeroCtr << "; nb basic columns: " << nbBasis << "\n";
            }

            delete env;
        }catch(GRBException e){
            std::cout << "Gurobi Exception\n";
            cout << e.getErrorCode() << " " << e.getMessage() << "\n";
        }
        
        return true;
    }


    void CG::record_training_data(string fpath){

        if (!lp_optimal) return;

        ofstream outf(fpath, ios_base::app);

        outf << setprecision(6) << dual_value[0];
        for (int i = 1; i < dual_value.size(); i++){
            outf << setprecision(6) << " " << dual_value[i];
        }
        outf << "\n";
        outf.flush();
        outf.close();
    }

    bool CG::solve_mwis_gurobi(double cutoff, double& min_rc) {

        if (cutoff <= 0)
            return false;

        // setup the model now
        GRBEnv *env;
        vector<GRBVar> x;
        env = new GRBEnv();
        GRBModel model = GRBModel(*env);
        model.set(GRB_DoubleParam_TimeLimit, cutoff);
        model.set(GRB_IntParam_Threads, thread_limit);
        model.getEnv().set(GRB_IntParam_OutputFlag, 0);
        model.set(GRB_StringAttr_ModelName, "MIP_MWIP");
        // Create variables and set them to be binary
        x.resize(g.size());
        for (int i = 0; i < g.size(); ++i){
            x[i] = model.addVar(0,1,0,GRB_BINARY);
        }
        model.update();

        // adjacent vertices cannot be selected simultaneously.
        for (int i = 0; i < g.size(); ++i){
            for (int j = i+1; j < g.size(); ++j){
                if (adj_matrix[i][j] == 1){
                    model.addConstr(x[i] + x[j] <= 1, "");
                }
            }
        }
        model.update();
        model.set(GRB_IntParam_Threads, thread_limit);
        // the objective
        GRBLinExpr tot = 1;
        for(int i = 0; i < g.size(); ++i){
            tot -= x[i] * dual_value[i];
        }
        model.setObjective(tot,GRB_MINIMIZE);
        model.update();
        auto t0 = get_wall_time();
        model.optimize();

        min_rc = model.get(GRB_DoubleAttr_ObjVal);
        bool mwis_optimal = (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);
        
        if (mwis_optimal){
            optimal_mis.clear();
            for (int i = 0; i < g.size(); ++i){
                if (x[i].get(GRB_DoubleAttr_X) > 0.5){
                    optimal_mis.push_back(i);
                }
            }
        }

        delete env;
        return mwis_optimal;
    }

}
