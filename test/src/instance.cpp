#include "instance.h"
#include <cmath>
#include <iterator>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include "util.h"
#include "primal.h"

namespace GCP {
    static bool is_file_exist(const char *fileName)
    {
        std::ifstream infile(fileName);
        return infile.good();
    };

    Instance::Instance(string inst_name, string data_dir, string dual_dir, int method, int seed) : 
        inst_name{inst_name}, data_dir{data_dir}, dual_dir{dual_dir}, method{method}, seed{seed} {
        read_graph();
        read_optimal_duals(true);
        init();
    }

    Instance::Instance(string inst_name, string data_dir, int seed) : inst_name{inst_name}, data_dir{data_dir}, seed{seed} {
        read_graph();
        hasOptimalDual = false;
    }

    void Instance::init(){
        
        // get lp columns
        int_cols.clear();
        runHeur(this, seed, int_cols);
        cout << method << "\n";
        string ml_prediction_dir = data_dir + "../dual_pred_";
        string ml_name="";
        estimateDual.resize(n_nodes);
        if (method == 0 || method == 1){
            for (int i = 0; i < n_nodes; i++){
                estimateDual[i] = 0;
            }
            return;
        }
        if (method == 2 || method == 3){ // degree
            if (output_level == 1) cout << "INFO: estimate dual using node degree." << endl;
            float max_nb_neighbor = n_nodes-1;
            for (int i = 0; i < n_nodes; i++){
                estimateDual[i] = degree[i] / max_nb_neighbor;
            }
        }else if  (method == 10 || method == 11){ // optimal dual solution
            assert(hasOptimalDual);
            estimateDual = optimalDualAvg;
        }else{ // ml predictions

            if (method == 4 || method == 5){
                ml_name =  "mlp_mse_3_identity";
            }else if (method == 6 || method == 7){
                ml_name =  "gcn_mse_20_identity";
            }else{
                assert(method == 8 || method == 9);
                ml_name =  "other_lr";
            }          

            ml_prediction_dir =  ml_prediction_dir + ml_name + "/";
            string fpath = ml_prediction_dir + inst_name + ".pred";
            ifstream infile(fpath);
            if (!infile.good()){
                cout << "cannot open dual prediction file: " << fpath << "\n";
                exit(1);
            }else{
                cout << "read prediction for the method: " << ml_name << "\n";
            }

            estimateDual.resize(n_nodes);
            string line;
            double dual_estimate;
            for (int i = 0; i < n_nodes; i++){
                getline(infile, line);
                stringstream stream(line);
                stream >> dual_estimate;
                estimateDual[i] = max(0., min(1., dual_estimate));
                if (degree[i]==0)
                    estimateDual[i] = 0;
                if (degree[i]==n_nodes-1)
                    estimateDual[i] = 1;
            }
            infile.close();
        }
    }

    void Instance::read_optimal_duals(bool multiple_avg){
        string dual_file = dual_dir  + inst_name + ".dual";
        ifstream file(dual_file);
        if (!file.good()){
            hasOptimalDual = false;
            return;
        }
        hasOptimalDual = true;

        string line;
        double dual;

        if (multiple_avg){
            while(!file.eof()) {
                getline(file, line);
                if (line.empty()) break;
                stringstream stream(line);
                vector<double> optimal_dual;
                for (int i = 0; i < n_nodes; i++){
                    stream >> dual;
                    optimal_dual.push_back(dual);
                }
                optimalDuals.push_back(optimal_dual);
            }
        }else{
            getline(file, line);
            stringstream stream(line);
            vector<double> optimal_dual;
            for (int i = 0; i < n_nodes; i++){
                stream >> dual;
                optimal_dual.push_back(dual);
            }
            optimalDuals.push_back(optimal_dual);
        }


        optimalDualAvg = vector<double> (n_nodes, 0);
        for (auto& optimalDual : optimalDuals){
            for (int i = 0; i < n_nodes; i++){
                optimalDualAvg[i] += optimalDual[i];
            }
        }
        for (int i = 0; i < n_nodes; i++){
            optimalDualAvg[i] /= optimalDuals.size();
            assert(optimalDualAvg[i] >= 0);
            assert(optimalDualAvg[i] <= 1);
        }
    }

    void Instance::read_graph(){
        string graph_file = data_dir  + inst_name + ".col";
        ifstream file(graph_file);
        if (!file.good()){
            cout << "cannot open data file: " << graph_file << "\n";
            successful = false;
        }
        successful = true;
        string line, s1, s2;
        int v1, v2, ne = 0;
        int idx;

        while(!file.eof()) {
            getline(file, line);

            if (line[0] == 'p') {

                stringstream stream(line);
                stream >> s1 >> s2 >> n_nodes >> n_edges;
                adj_list.resize(n_nodes);
                density = (2*n_edges) / (double)(n_nodes * (n_nodes-1));
                if (output_level == 1){
                    cout << "INFO: number of nodes is " << n_nodes << "\n";
                    cout << "INFO: number of edges is " << n_edges << "\n";\
                    cout << setprecision(2) << "INFO: density is " << density << "\n"; 
                }
                if (n_edges == 0) return;               
            }
            if (line[0] == 'e'){

                // vector<string> tokens;
                // boost::split(tokens, line, boost::is_any_of(" "));
                // if (tokens.size() != 3) {
                //     continue;
                // }

                stringstream stream(line);
                stream >> s1 >> v1 >> v2;
                if (v1 == v2) continue;
                if (find(adj_list[v1-1].begin(), adj_list[v1-1].end(), v2-1) == adj_list[v1-1].end()){
                    adj_list[v1-1].push_back(v2-1);
                    adj_list[v2-1].push_back(v1-1);
                    ne++;
                }
                // for (idx = 0; idx < adj_list[v1-1].size(); ++idx){
                //     if (adj_list[v1-1][idx] == v2 - 1){
                //         break;
                //     }
                // }
                // if (idx == adj_list[v1-1].size()){
                //     adj_list[v1-1].push_back(v2-1);
                //     adj_list[v2-1].push_back(v1-1);
                //     ne++;
                // }
            }
        }

        degree_norm = vector<double>(n_nodes);
        degree = vector<int>(n_nodes);
        max_node_degree_norm = 0.0;
        max_node_degree = 0.;

        for (int i = 0; i < n_nodes; ++i){

            sort(adj_list[i].begin(), adj_list[i].end());

            degree[i] = adj_list[i].size();
            degree_norm[i] = (double) degree[i]/ (double) n_nodes;
            if (max_node_degree < degree[i]){
                max_node_degree = degree[i];
                max_node_degree_norm = degree_norm[i];
            }
        }
        n_edges = ne;


        // adj matrix
        adj_matrix = vector<vector<bool>>(n_nodes, vector<bool>(n_nodes, 0));
        for (int i = 0; i < n_nodes; ++i){
            for (int j = 0; j < adj_list[i].size(); ++j){
                adj_matrix[i][adj_list[i][j]] = 1;
                adj_matrix[adj_list[i][j]][i] = 1;
            }
        }
    }

    int* Instance::getFirstAdjedge(int v){
        return &adj_list[v][0];
    }

    int* Instance::getLastAdjedge(int v){
        return &adj_list[v][adj_list[v].size()-1];
    }


    bool Instance::is_edge(int v1, int v2){
        return adj_matrix[v1][v2] == 1;
    }

    double Instance::compute_and_record_features(string feature_dir, int sample_factor){

        auto t0 = get_cpu_time();
        vector<vector<int>> col_set; set<string> col_identities;        
        gen_random_cols(adj_matrix, current_time_for_seeding(), sample_factor * n_nodes, col_set, col_identities);
        double nb_col = col_set.size();
        // vector<vector<bool>> col_set_binary(nb_col, vector<bool>(n_nodes, 0));
        // long v;
        // for (long i = 0; i < nb_col; ++i){
        //     for (long j = 0; j < col_set[i].size(); ++j){
        //         v = col_set[i][j];
        //         col_set_binary[i][v] = 1;
        //     }
        // }


        int nb_feat = 9; 
        vector<vector<double>> featM(n_nodes, vector<double>(nb_feat, 0.));
        vector<double> nb_appears(n_nodes, 0);

        // feature: degree (0)
        for (int i = 0; i < n_nodes; i++){
            featM[i][0] = degree[i];
        }

        // feature: frequency in sample(1)
        for (int i = 0; i < nb_col; i++){
            for (int j = 0; j < col_set[i].size(); j++){
                int cur_node = col_set[i][j];
                nb_appears[cur_node] += 1;
            }
        }
        for (int i = 0; i < n_nodes; i++){
            featM[i][1] = nb_appears[i] / nb_col;
        }

        // features: max (2), min (3), and avg (4) cardinality of samples containing a vertex
        for (int i = 0; i < n_nodes; i++){
            featM[i][3] = 1e8;
        }

        
        for (int i = 0; i < nb_col; i++){
            double cardinality = col_set[i].size();
            for (int v : col_set[i]){
                featM[v][2] = max(featM[v][2], cardinality);
                featM[v][3] = min(featM[v][3], cardinality);
                featM[v][4] +=cardinality;
            }
        }
        for (int i = 0; i < n_nodes; i++){
            featM[i][4] /= nb_appears[i];
        }

        // features: max (5), min (6), and avg (7) average node degree in samples containing a vertex

        vector<double> avg_degrees(nb_col, 0);
        for (int i = 0; i < nb_col; i++){
            for (int v : col_set[i]){
                avg_degrees[i] += degree[v];
            }
        }
        for (int i = 0; i < nb_col; i++){
            avg_degrees[i] /= col_set[i].size();
        }

        for (int i = 0; i < n_nodes; i++){
            featM[i][6] = 1e8;
        }

        for (int i = 0; i < nb_col; i++){
            double avg_deg = avg_degrees[i];
            for (int v : col_set[i]){
                featM[v][5] = max(featM[v][5], avg_deg);
                featM[v][6] = min(featM[v][6], avg_deg);
                featM[v][7] += avg_deg;
            }
        }        
        for (int i = 0; i < n_nodes; i++){
            featM[i][7] /= nb_appears[i];
        }
        
        // feature: density (8)
        for (int i = 0; i < n_nodes; i++){
            featM[i][8] = density;
        }    

        // normalize
        for (int feat_idx = 0; feat_idx < nb_feat-1; feat_idx++){
            double max_val = -1e8;
            for (int vertex_idx = 0; vertex_idx < n_nodes; vertex_idx++){
                max_val = max(max_val, featM[vertex_idx][feat_idx]);
            }

            for (int vertex_idx = 0; vertex_idx < n_nodes; vertex_idx++){
                featM[vertex_idx][feat_idx] = featM[vertex_idx][feat_idx] / max_val;
            }
        }
        auto t = get_cpu_time() - t0;
        

        // for (int i = 0; i < 20; i++){
        //     cout << i << ": ";
        //     for (int j = 0; j < nb_feat; j++){
        //         cout << featM[i][j] << " ";
        //     }
        //     cout << "\n";
        // }
        // exit(1);

        ofstream feat_file(feature_dir + "/" + inst_name + ".feat");
        for (int vertex_idx = 0; vertex_idx < n_nodes; vertex_idx++){
            for (int feat_idx = 0; feat_idx < nb_feat; feat_idx++){
                feat_file << featM[vertex_idx][feat_idx] << " ";
            }
            feat_file << "\n";
        }

        feat_file.flush();
        feat_file.close();
        return t;
    }
}