#include "instance.h"
#include <boost/filesystem.hpp>
#include <omp.h>
#include "util.h"
#include "CG.h"
using namespace GCP;
using namespace std;

void collect_label(){

    double cutoff_time=100;
    double cutoff_time_pricer=10;

    omp_set_num_threads(32);
    int pricing_method = 2;
    int column_selection = 0; 
    string line;

    srand(1);
    std::random_device rd;
    std::default_random_engine rng(rd());

    string input_dir = "../../data/instances/";
    string output_dir = "../../data/dual_opt/";
    boost::filesystem::create_directories(output_dir);

    vector<int> degenerate_inst_indices;
    vector<int> degenerate_ctr;

    int nb_inst = 8278;
    vector<int> inst_ids;
    for (int i = 1; i <= nb_inst; i++){
        inst_ids.push_back(i);
    }
    shuffle(inst_ids.begin(), inst_ids.end(), rng);

    int nb_solved = 0;
    #pragma omp parallel for
    for (int i = 0; i < nb_inst; i++){
        
        string inst_name = "g" + ToString(inst_ids[i], 4);
        auto instance = Instance(inst_name, input_dir);

        if (instance.get_nb_edges()==0 || !instance.successful){
            #pragma omp critical
            {
                cout << "ERROR: cannot read: " << inst_name << "\n";
            }
        }else{
            auto cg = CG(instance, pricing_method, cutoff_time, cutoff_time_pricer, 1, 1);
            cg.solve(nullptr);
            if (cg.lp_optimal){
                string output_file = output_dir + inst_name + ".dual";
                cg.record_training_data(output_file);
                #pragma omp critical
                {
                    nb_solved++;
                    if (cg.degenerateCtr > 0){
                        degenerate_inst_indices.push_back(inst_ids[i]);
                        degenerate_ctr.push_back(cg.degenerateCtr);
                    }
                }
            }
        }
    }


    // ofstream outf("../degenerate_data.txt");
    // for (int i = 0; i < degenerate_inst_indices.size(); i++){
    //     outf << degenerate_inst_indices[i] << " " << degenerate_ctr[i] << "\n";
    // }
    
    cout << "INFO: nb of optimally solved instances (within 100s): " << nb_solved << "\n";

    vector<pair<int, int>> pairs;
    for (int i = 0; i < degenerate_inst_indices.size(); i++){
        int nb_run = degenerate_ctr[i] * 10;
        for (int j = 0; j < nb_run; j++){
            pairs.push_back(pair<int, int> (degenerate_inst_indices[i], j+10000));
        }
    }
    cout << "INFO: solve "<< pairs.size() / 10 << " degenerated instances, total number of runs: " << pairs.size();
    shuffle(pairs.begin(), pairs.end(), rng);

    #pragma omp parallel for
    for (int i = 0; i < pairs.size(); i++){
        string inst_name = "g" + ToString(pairs[i].first, 4);
        auto instance = Instance(inst_name, input_dir);
        if (!instance.successful){
            #pragma omp critical
            {
                cout << "cannot read: " << inst_name << "\n";
            }
        }else{
            auto cg = CG(instance, pricing_method, cutoff_time, cutoff_time_pricer, 1, pairs[i].second);
            cg.solve(nullptr);
            if (cg.lp_optimal){
                string output_file = output_dir + inst_name + ".dual";
                #pragma omp critical
                {
                    cg.record_training_data(output_file);
                }
            }
        }
    }
}

void collect_feature(){

    int sample_factor = 5;
    const string data_dir = "../../data/instances/";
    const string output_dir = "../../data/features/";
    boost::filesystem::create_directories(output_dir);

    vector<string> file_names;
    string fpath = "../../data/lists/all.txt";
    ifstream infile(fpath);
    if (!infile.good()){
        cout << "cannot find file: " << fpath << "\n";
    }

    string line;
    while(!infile.eof()) {
        getline(infile, line);
        if (line.empty()) break;
        file_names.push_back(line);
    }

    if (output_level == 1) cout << "total number of instances: " << file_names.size() << "\n";
    omp_set_num_threads(8);
    #pragma omp parallel for
    for (int i = 0; i < file_names.size(); i++){
        string inst_name = file_names[i];
        auto instance = Instance(inst_name, data_dir);
        instance.compute_and_record_features(output_dir, sample_factor);
    }
}

int main(int argc, char* argv[]) {


    // collect_label();
    // collect_feature();
 
    return 0;
}
