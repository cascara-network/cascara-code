# Cascara
This repository has APIs needed to run Cascara and its variants given traffic demands, link costs and topology information as input. We implemented Cascara using both CVXPY with GUROBI as the solver and native gurobipy API bindings. 

* The native implementation is in `api_opt_native.py` and the `cvxpy` version is in `api_native.py`. The native implementation gave us control over setting parameters of GUROBI. 

* Most likely the set of parameters that give you the fastest results from the solver will be slightly different from ours. This is highly dependent on your traffic demands and link capacities.
* `solve_optimization` is the main function in both versions. It takes demands as a dictionary along with other details about the network. It sets up the problem formulation with relevant constraints and objective. Once solutions are found, it dumps them into a `CSV`.

* We have removed all references to our input files so those variables will not be available. Replace them with your inputs.

## Notes on Cascara-online
* The `heuristic_api.py` has all the functionality required for Cascara-online, Entact, GFA and related algorithms.
* For confidentiality reasons we have removed the `alpha` and `beta` values from the code. These can be found by a parameter sweep. Check the technical report for details.
