#include "CG.h"
#include "MLPH.h"
#include <iostream>
#include <cassert>
#include <iostream>
#include "pricer.h"
#include "MWCP_solver.h"
#include "mwcq_solvers/lscc.hpp"

namespace GCP {
using namespace std;

    CG::CG(Instance& g, double epsilon, int seed, double cutoff, int thread_limit): 
            g{g}, epsilon{epsilon}, cutoff{cutoff}, seed{seed}, thread_limit{thread_limit} {
        lp_method = -1;
        rmp_obj = 1e8;
        rmp_obj_pre = 1e8;
        adj_list = g.adj_list;
        adj_matrix = g.adj_matrix;
        dual_value.resize(g.size());
        fill(dual_value.begin(),dual_value.end(),0);
    }

    void CG::solve(bool use_heuristic){
        
        col_set = g.int_cols;
        double start_time = get_wall_time();
        lp_optimal = false;
        min_reduced_cost = -1.0;
        bool mwis_optimal = true;;
        cg_iters = 0;
        double current_time;
        double cur_rmp_duration;
        double cur_mwis_time;
        dual_value = g.estimateDual;
        int method = g.method;

        while(true){
            cg_iters++; 

            current_time = get_wall_time(); 
            
            // scg requires previous dual iterate, so do not use scg at the initial iteration
            if (cg_iters==1){
                rmp_dual(0, 1e8);
            }else{
                rmp_dual(method, 1e8);
            }

            lps.push_back(rmp_obj);

            if (cg_iters == 1) rmp_obj_first = rmp_obj;
            cur_rmp_duration = get_wall_time()-current_time;
            lptime += cur_rmp_duration;

            current_time = get_wall_time();
            auto mwis_cutoff = cutoff - (current_time - start_time);
            mwis_optimal = false;

            if (mwis_cutoff <= 0) break;
            
            if (!use_heuristic){
                mwis_optimal = solve_mwis_tsm(mwis_cutoff, min_reduced_cost);
            }else{
                solve_mwis_lscc(mwis_cutoff, min_reduced_cost);

                if (min_reduced_cost > -1e-6 && method == 0){
                    mwis_optimal = solve_mwis_tsm(mwis_cutoff, min_reduced_cost);
                }
            }
        
            cur_mwis_time = get_wall_time()-current_time;
            pricingtime += cur_mwis_time;
            col_set.push_back(best_mis); // add new columns
            
            lp_optimal = mwis_optimal && method==0 && min_reduced_cost > -1e-6;
            if (lp_optimal) break;
            if (get_wall_time()-start_time >= cutoff) break;


            if (method != 0){
                if (method == 1 || method == 2 || method == 4 || method == 6 || method ==8 || method==10){ // scg
                    if (min_reduced_cost > -1e-6)
                        epsilon/=2;

                }else{ // ascg
                    epsilon = min_reduced_cost > -1e-6 ? 0 : min_reduced_cost / (min_reduced_cost-1);
                }

                if (epsilon < 1e-2){
                    method=0;
                    epsilon=0;
                }
            }
        }
    }


    bool CG::rmp_dual(int method, double lp_cutoff) {
        int nb_col = col_set.size();
        // setup the model now
        try{
            GRBEnv *env;
            env = new GRBEnv();
            GRBModel model = GRBModel(*env);
            model.set(GRB_IntParam_OutputFlag, 0);
            model.set(GRB_IntParam_Threads, thread_limit);
            model.set(GRB_StringAttr_ModelName, "RMP_GCP_dual");
            model.set(GRB_IntParam_Method, 0);
            model.set(GRB_DoubleParam_TimeLimit, lp_cutoff);

            // Create dual variables
            vector<GRBVar> lambda;
            vector<GRBVar> posvio;
            vector<GRBVar> negvio;

            lambda.resize(g.size());
            posvio.resize(g.size());
            negvio.resize(g.size());

            for (long i = 0; i < g.size(); ++i){
                lambda[i] = model.addVar(0,1,0,GRB_CONTINUOUS);
                posvio[i] = model.addVar(0,1,0,GRB_CONTINUOUS);
                negvio[i] = model.addVar(0,1,0,GRB_CONTINUOUS);
            }

            // each vertex is covered by at least one set
            vector<GRBConstr> S;
            S.resize(nb_col);
            for (long j = 0; j < nb_col; ++j){
                GRBLinExpr rtot = 0;
                for (auto v_id : col_set[j]){
                    rtot += lambda[v_id] ;
                }
                S[j] = model.addConstr(rtot <= 1, "");
            }


            vector<GRBConstr> R;
            R.resize(g.size() * 2);
            
            for (long k = 0; k < g.size(); k++){
                if (method == 0){

                }
                else if (method == 1){
                    R[k] = model.addConstr(lambda[k] - posvio[k] <= dual_value[k], "");
                    R[k+g.size()] = model.addConstr(-lambda[k] - negvio[k] <= -dual_value[k], "");
                }else{
                    R[k] = model.addConstr(lambda[k] - posvio[k] <= g.estimateDual[k], "");
                    R[k+g.size()] = model.addConstr(-lambda[k] - negvio[k] <= -g.estimateDual[k], "");
                }
            }
            
            model.update();
            // the objective
            GRBLinExpr tot=0;
            for(long i = 0; i < g.size(); ++i){
                if (method==0)
                    tot+=lambda[i];
                else
                    tot += lambda[i] - epsilon * posvio[i] - epsilon * negvio[i];
            }
            model.setObjective(tot,GRB_MAXIMIZE);
            model.update();
            // cout << "LP solver: " << model.get(GRB_IntParam_Method) << "\n";
            model.optimize();

            // 2 - OPTIMAL
            // 3 - INFEASIBLE
            // cout << "INFO: RMP solving status: " << model.get(GRB_IntAttr_Status) << "\n";
            assert(model.get(GRB_IntAttr_Status)==GRB_OPTIMAL);

            // get RMP solving statistics

            rmp_obj_pre = rmp_obj;
            rmp_obj = 0;
            // get optimal dual
            for (long j = 0; j < g.size(); ++j){
                dual_value[j] = lambda[j].get(GRB_DoubleAttr_X);
                rmp_obj += dual_value[j];
            }
            // assert(isEqual(rmp_obj, model.get(GRB_DoubleAttr_ObjVal)));
            // rmp_obj = model.get(GRB_DoubleAttr_ObjVal);    
            // assert(!isGreater(rmp_obj, rmp_obj_pre));

            delete env;
        }catch(GRBException e){
            std::cout << "Gurobi Exception\n";
            cout << e.getErrorCode() << " " << e.getMessage() << "\n";
        }
        
        return true;
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
        auto current_time = get_wall_time();
        model.optimize();

        min_rc = model.get(GRB_DoubleAttr_ObjVal);
        bool mwis_optimal = (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL);
        
        if (mwis_optimal){
            best_mis.clear();
            for (int i = 0; i < g.size(); ++i){
                if (x[i].get(GRB_DoubleAttr_X) > 0.5){
                    best_mis.push_back(i);
                }
            }
        }

        delete env;
        return mwis_optimal;
    }



    bool CG::solve_mwis_lscc(double cutoff, double& min_rc){
        if (cutoff <= 0)
            return false;
        
        int nb_node = g.size();
        int nb_edge = (nb_node*(nb_node-1))/2. - g.get_nb_edges();
        long long** AdjacentList;
        long long* Node_Degree;
        long long* Node_Weight;
        Node_Weight = (long long *) malloc((nb_node) * sizeof(long long));
        for (auto i = 0; i < nb_node; ++i){
            Node_Weight[i] = dual_value[i] * 1e12;
        }

        Node_Degree = (long long *) malloc((nb_node) * sizeof(long long));
        AdjacentList = (long long **) malloc((nb_node) * sizeof(long long *));
        
        for (auto i = 0; i < nb_node; i++){
            Node_Degree[i] = nb_node - adj_list[i].size() - 1;
            AdjacentList[i] = (long long *) malloc(Node_Degree[i] * sizeof(long long));
            vector<bool> candidates(nb_node, true);
            candidates[i]=false;
            for (auto j = 0; j < adj_list[i].size(); j++){
                candidates[adj_list[i][j]] = false;
            }
            auto k = 0;
            for (auto j = 0; j < nb_node; j++){
                if(candidates[j]){
                    AdjacentList[i][k++] = j;
                }
            }
            // std::cout << i << " " << k << " " << Node_Degree[i] << "\n";
            if (k!=Node_Degree[i]) 
                cout << "ERROR k!= Node_Degree in constructer MWCP_solver.cpp\n";
            assert(k==Node_Degree[i]);
        }



        LSCC::LSCC solver;
        double best_obj = solver.lscc(nb_node, nb_edge, cutoff, AdjacentList, Node_Degree, Node_Weight, seed);
        assert(best_obj == solver.sol_objs[solver.sol_objs.size()-1]);

        free(Node_Degree);
        free(Node_Weight);
        for (auto i = 0; i < nb_node; i++)
            free(AdjacentList[i]);
        free(AdjacentList);

        min_rc = 1.;
        best_mis =  solver.sols[ solver.sols.size()-1];
        for (auto v : best_mis){
            min_rc -= dual_value[v];
        }
        return false;
    }

    bool CG::solve_mwis_tsm(double cutoff, double& min_rc){
        if (cutoff <= 0)
            return false;
        
        // for (int i = 0; i < g.size(); i++){
        //     assert(dual_value[i] >= 0);
        //     assert(dual_value[i] <= 1);
        // }
        MWCP_solver tsm(8, cutoff, dual_value, adj_list, adj_matrix, g.get_nb_edges(),1e8);
        tsm.run();
        best_mis=tsm.optimal_mis;
        min_rc=tsm.exact_rc;
        
        return tsm.isOptimal;
    }
}
