import sys, os, time, subprocess
from math import floor
import numpy as np

CG=0; SCG=1; 
SCG_DEG=2; ASCG_DEG=3
SCG_FFNN=4; ASCG_FFNN=5
SCG_GCN=6; ASCG_GCN=7

def gmean(arr, shift=10):
    arr = np.array(arr, dtype=float)
    log_a = np.log(arr+shift)
    return np.abs(np.exp(log_a.mean(axis=0)) - shift)

def test_matilda(test_names, seeds, methods, nCPU = 24):

    EXACT = 0
    
    tuples = []
    for seed in seeds:
        for test_name in test_names:
            for method in methods:
                tuples.append((test_name, method[0], method[1], seed))

    nb_run = len(tuples)
    print(f"number of runs: {nb_run}")
    tuples = (p for p in tuples)
    # exit(1)

    proc_pool = []
    start_time = time.time()
    nb_finished = 0    
    duration = 0
    hasNext = True

    while hasNext or len(proc_pool) > 0:
        # remove fininshed processes
        for idx in reversed(range(len(proc_pool))):
            proc, item = proc_pool[idx]
            if proc.poll() is not None:
                if proc.returncode != 0:
                    errorMsg = ','.join([str(tok) for tok in item])
                    print(f'error in executing: {errorMsg}')
                del proc_pool[idx]
                nb_finished+=1

        # execute new runs
        while len(proc_pool) < nCPU and hasNext:
            item = next(tuples,None)
            hasNext = item is not None
            if hasNext:
                (test_name, method, penalty, seed) = item
                proc = subprocess.Popen(f'./CG {test_name} {method} {EXACT} {penalty} {seed} 1> /dev/null', cwd=f'./build', shell=True)
                proc_pool.append((proc, item))
                
        # report progress
        curr_time = time.time()
        cur_duration = floor((curr_time - start_time)/60)
        if (cur_duration > duration):
            duration = cur_duration
            print(f"time used: {duration} minutes\nnumber of finished runs: {nb_finished}\nnumber of remaining runs: {nb_run - nb_finished}\n")
            sys.stdout.flush()
        time.sleep(0.01)


def result_stats(test_names, seeds, methods):
    
    # dual estimate options
    convert = {CG:'cg', SCG:'scg', SCG_DEG:'scg-deg', ASCG_DEG:'ascg-deg',
                SCG_FFNN:'scg-ffnn', ASCG_FFNN:'ascg-ffnn',
                SCG_GCN:'scg-gcn', ASCG_GCN:'ascg-gcn'}
    
    for i in range(len(methods)):
        methods[i] = (convert[methods[i][0]], methods[i][1])

    # preprocess    
    for (m, penalty) in methods:
        writeto = f"{m}.txt"
        with open(writeto, 'w') as fw:
            fw.write('inst_name,seed,optimality,obj,obj_first,#CG_iter,wall_time,cpu_time,lptime,pricingtime\n')
            for test_name in test_names:
                for seed in seeds:
                    ret_path = f'results/exact_{m}_{penalty}_{seed}/{test_name}.ret'
                    with open(ret_path, 'r') as fr:
                        line = fr.readlines()[1].strip()
                    fw.write(f'{test_name},{seed},{line}\n')


    # load result
    method_ret_dict = {}

    for (m, penalty) in methods:
        if m not in method_ret_dict:
            method_ret_dict[m] = {}
        
        ret_fpath=f'{m}.txt'

        with open(ret_fpath, 'r') as f:
            lines = f.readlines()[1:]
        for line in lines:
            inst_name,seed,optimality,obj,obj_first,nCG_iter,wall_time,cpu_time,lptime,pricingtime = line.strip().split(',')
            seed = int(seed)
            if seed > 10:
                continue
            if 'opt_ctr' not in method_ret_dict[m]:
                method_ret_dict[m]['opt_ctr'] = int(optimality)
            else:
                method_ret_dict[m]['opt_ctr'] += int(optimality)

            if inst_name not in method_ret_dict[m]:
                method_ret_dict[m][inst_name] = [float(wall_time), float(lptime), float(pricingtime),
                     float(nCG_iter), float(obj_first), 1]
            else:
                method_ret_dict[m][inst_name][0] += float(wall_time)    
                method_ret_dict[m][inst_name][1] += float(lptime)          
                method_ret_dict[m][inst_name][2] += float(pricingtime)
                method_ret_dict[m][inst_name][3] += float(nCG_iter)
                method_ret_dict[m][inst_name][4] += float(obj_first)
                method_ret_dict[m][inst_name][5] += 1 # count number of seeded runs

    for inst_name in test_names:
        for (m,penalty) in methods:
            assert(inst_name in method_ret_dict[m])
            method_ret_dict[m][inst_name][3] /= method_ret_dict[m][inst_name][5]
            method_ret_dict[m][inst_name][4] /= method_ret_dict[m][inst_name][5]

    # compute mean statistics
    write_to = 'stats'    
    d = {}
    with open(f'{write_to}.txt', 'w') as f:
        f.write('method,niter,tottime\n')
        print('method,niter,tottime')
        for i, m in enumerate(method_ret_dict.keys()):
            tot_wall_time = 0
            tot_lp_time = 0
            tot_pricing_time = 0
            all_iter = []
            tot_iter = 0
            all_opt_ctr = method_ret_dict[m]['opt_ctr']
            for inst_name in test_names:
                
                if inst_name not in d:
                    d[inst_name]=[]
                d[inst_name].append(int(method_ret_dict[m][inst_name][0]))

                # if inst_name not in method_ret_dict[estimate_method]:
                #     continue
                tot_wall_time+=method_ret_dict[m][inst_name][0]
                tot_lp_time += method_ret_dict[m][inst_name][1]
                tot_pricing_time +=method_ret_dict[m][inst_name][2]
                avg_iter = method_ret_dict[m][inst_name][3]
                all_iter.append(avg_iter)
                tot_iter += method_ret_dict[m][inst_name][3]
            avg_all_iter = gmean(all_iter)
            # avg_all_iter = np.mean(all_iter)

            f.write(f'{m},{round(avg_all_iter,1)},{int(tot_wall_time)}\n')
            print(f'{m},{round(avg_all_iter,1)},{int(tot_wall_time)}')
    os.system(f'python3 tably.py -e {write_to}.txt > {write_to}.tex')
    os.remove(f'{write_to}.txt')


def get_test_names(test_list_file_path):
    with open(test_list_file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if len(line)!= 0]

if __name__ == '__main__':

    os.system(f'mkdir build; cd build && cmake ../ && make')

    fpath = "../data/lists/test.txt"
    test_insts = get_test_names(fpath)
    seeds = [i for i in range(1, 11)]
    
    methods = [(CG, 0), (SCG, 1), (SCG_DEG, 0.1), (SCG_FFNN, 0.1),
               (SCG_GCN, 0.1), (ASCG_FFNN, 1), (ASCG_GCN, 1)]
    
    # test_matilda(test_insts, seeds, methods)
    result_stats(test_insts, seeds, methods)