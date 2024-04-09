#ifndef UTIL_H
#define UTIL_H


#include <limits>
#include <chrono>
#include <cmath>
#include <math.h>
#include <iterator>
#include <iostream>
#include <numeric>      // std::iota
#include <random>
#include <algorithm>
#include <vector>
#include <cstring>
#include <string>
#include <time.h>
#include <sys/time.h>
#include <iomanip>
#include <sstream>
#include <utility>
#include <set>


namespace GCP{
using namespace std;

    // 0: no output; 
    // 1: output 
    const int output_level = 0;
    const double eps = 1e-6;

    inline double isEqual(double d1, double d2){
        return abs(d1 - d2) < eps;
    }

    inline double isGreater(double d1, double d2){
        return d1 - d2 > eps;
    }

    inline double isSmaller(double d1, double d2){
        return d2 - d1 > eps;
    }

    inline double get_wall_time(){
        struct timeval time;
        if (gettimeofday(&time,NULL)){
            return 0;
        }
        return (double)time.tv_sec + (double)time.tv_usec * .000001;
    }

    inline long current_time_for_seeding(){
        using namespace chrono;
        long now = duration_cast< nanoseconds >(
        system_clock::now().time_since_epoch()).count();
        return now;
    }

    inline string ToString(int value, int digitsCount){
        ostringstream os;
        os << setfill('0') << setw(digitsCount) << value;
        return os.str();
    }

    inline double get_cpu_time(){
        return (double)clock() / CLOCKS_PER_SEC;
    }

    inline vector<long> sort_indexes_inc(const vector<double> &v) {
        // initialize original index locations
        vector<long> idx(v.size());
        iota(idx.begin(), idx.end(), 0);
        stable_sort(idx.begin(), idx.end(),
            [&v](long i1, long i2) {return v[i1] < v[i2];});

        return idx;
    }

    inline vector<long> sort_indexes_dec(const vector<double> &v) {
        // initialize original index locations
        vector<long> idx(v.size());
        iota(idx.begin(), idx.end(), 0);
        stable_sort(idx.begin(), idx.end(),
            [&v](long i1, long i2) {return v[i1] > v[i2];});
        return idx;
    }


    void gen_random_cols(vector<vector<bool>>& adj_matrix, int seed, int nb_col, vector<vector<int>>& cols, set<string>& col_identities);

    inline void sort_indexes_noninc(int* values, int* sortednodes, int len) {

        vector<double> val(len);
        for (int i = 0; i < len; i++){
            val[i] = values[i];
        }
        vector<long> idx = sort_indexes_dec(val);

        vector<int> sorted(len);

        for (int i = 0; i < len; i++){
            sorted[i] = sortednodes[idx[i]];
            values[i] = val[idx[i]];
        }

        for (int i = 0; i < len; i++){
            sortednodes[i] = sorted[i];
        }
    }

}

#endif