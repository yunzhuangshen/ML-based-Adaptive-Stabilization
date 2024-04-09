#ifndef INSTANCE_H
#define INSTANCE_H
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <sstream>
#include <cmath>
#include <random>
#include <iterator>
#include <assert.h>
using namespace std;

// extern "C" {
// #include <igraph.h>
// }

namespace GCP {
    class Instance {
        const string data_dir;
        const string dual_dir;
        void read_graph();
        vector<vector<int>> allMISs;
        vector<vector<int>> node_to_sets;

    public:
        int n_nodes;
        int n_edges;
        int method;
        double density;
        int nMIS;
        vector<vector<int>> int_cols;
        vector<vector<double>> optimalDuals;
        vector<double> optimalDualAvg;
        vector<double> estimateDual;
        const string inst_name;
        vector<vector<int>> adj_list;
        vector<vector<bool>> adj_matrix;
        double max_node_degree_norm;
        double max_node_degree;
        vector<int> degree;
        vector<double> degree_norm;
        bool successful;
        bool hasOptimalDual;

        int seed = 1;
        // Created a new (random) graph with n_nodes nodes
        Instance(string inst_name, string data_dir, string dual_dir, int method, int seed);
        Instance(string inst_name, string data_dir, int seed);
        // Size of the graph (i.e. number of nodes).
        int size() const { return n_nodes; }
        int get_num_edges() const { return n_nodes; }
        vector<vector<int>> get_adj_list() const {return adj_list; }
        // Optimal solution of edge x[i][j]
        string get_inst_name() const { return inst_name; }
        void enumerate_all_MISs();
        void read_optimal_duals(bool multiple_avg);
        int get_nb_edges() const { return n_edges; }
        double compute_and_record_features(string feature_dir, int sample_factor);
        bool is_edge(int v1, int v2);
        void init();
        int* getFirstAdjedge(int v);
        int* getLastAdjedge(int v);
    };
}

#endif
