#include "instance.h"
#include "CG.h"
#include <boost/filesystem.hpp>
#include <omp.h>
#include "util.h"

using namespace GCP;
using namespace std;


// (test_name, method, penalty, seed)
void test(string inst_name, int method, bool use_heuristic, double penalty, int seed){


    double cutoff_time=1e8;
    int thread_limit=1;

    const string data_dir = "../../data/instances/";
    const string dual_dir = "../../data/dual_opt_processed/";
    
    string str_method;

    switch (method)
    {
    case 0: str_method="cg";break;
    case 1: str_method="scg";break;
    case 2: str_method="scg-deg";break;
    case 3: str_method="ascg-deg";break;
    case 4: str_method="scg-ffnn";break;
    case 5: str_method="ascg-ffnn";break;
    case 6: str_method="scg-gcn";break;
    case 7: str_method="ascg-gcn";break;

    default:
        cout << "dual estimate method not exists: " << method << "\n";
        exit(-1);
    }

    string str_penalty=to_string(penalty);
    str_penalty.erase ( str_penalty.find_last_not_of('0') + 1, std::string::npos );
    str_penalty.erase ( str_penalty.find_last_not_of('.') + 1, std::string::npos );

    string heur = use_heuristic ? "heur" : "exact";
    string output_dir = "../results/" + heur + "_" + str_method + "_" + str_penalty + "_" + to_string(seed) + "/";
    boost::filesystem::create_directories(output_dir);
    
    auto instance = Instance(inst_name, data_dir, dual_dir, method, seed);
    string output_solving_filename;
    output_solving_filename = output_dir + inst_name + ".ret";
    ofstream output_file_solving_stats (output_solving_filename);

    if (output_file_solving_stats.is_open()){
        output_file_solving_stats << "optimality,obj,obj_first,#CG_iter,walltime,cputime,lptime,pricingtime" << endl;
    } else{
        cout << "Cannot open the output file " + output_solving_filename << endl;
        // return 0;
    }

    auto lpobj_filename = output_dir + inst_name + ".objs";
    ofstream lpobj_file (lpobj_filename);

    auto cg = CG(instance, penalty, seed, cutoff_time, thread_limit);
    cout << "SOLVING ROOT LP BY CG\n"; 
    auto w0 = get_wall_time();
    auto c0 = get_cpu_time();
    cg.solve(use_heuristic);
    auto cpu_time_CG = get_cpu_time()-c0;
    auto wall_time_CG = get_wall_time() - w0;

    if (cg.lp_optimal){
        cout << "ROOT LP SOLVED.\n time used: " << wall_time_CG << "s\niteration used: " << cg.cg_iters << "\n";
    }else{
        cout << "ROOT LP UNSOLVED\n";
    }

    output_file_solving_stats << cg.lp_optimal << ","  << cg.rmp_obj << "," << cg.rmp_obj_first << ","
                                << cg.cg_iters << "," << wall_time_CG << "," << cpu_time_CG << ","
                                << cg.lptime << "," << cg.pricingtime << "\n";
    
    output_file_solving_stats.close();

    for (auto tmp : cg.lps){ 
        lpobj_file << tmp << "\n";
    }
    lpobj_file.close();
}


int main(int argc, char* argv[]) {


    // (test_name, method, use_heuristic, penalty, seed)
        test(argv[1], stoi(argv[2]), stoi(argv[3]), stod(argv[4]), stoi(argv[5]));
    
    // test("g0330", 0, 2, 0, 1, 0, 1); 89 iteration


    return 0;
}
