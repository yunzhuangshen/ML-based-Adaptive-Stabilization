
# **Adaptive Stabilization Based on Machine Learning for Column Generation**

# Structure
- **test** folder contains the C++ code for testing the proposed ASCG method.
- **data** folder contains the Matilda test graphs and ML predictions. 
- **collect_data** folder contains the C++ code for collecting training data.
- **ml** folder contains the code for training ML models. You will need to download the full Matilda graph library at The whole Matilda benchmark can be downloaded at https://matilda.unimelb.edu.au/matilda/matildadata/graph_coloring_problem/2014/instances/graphs.zip.

# Requirements:
- Python 3.9: 
- C++ boost, openmp, Gurobi version 11

# Procedure to reproduce results in Matilda
- Run the command **python3 run_test.py** under **test** folder.