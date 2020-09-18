import time
import os
import json
from datetime import datetime
import pandas as pd
from consts import *
import pdb
import csv
import numpy as np
from cvxpy import *

def initialize_opti_vars_previous_work(flow_values, router_set):
    '''
    The difference between this initialization of optimization
    variables and and the one in 'initialize_opti_vars' is that
    this assigns a vector of variables for each router and every
    index of the router vector is for a timeslot in the month.
    These per-router vectors allow the use of built-in convex
    functions like 'sum_largest' which only accept vectors and not
    lists of scalars.
    https://www.cvxpy.org/tutorial/functions/index.html
    '''
    num_vars = len(flow_values)
    rtr_to_var_vector = {}
    for rtr in router_set:
        rtr_var = Variable(num_vars)
        rtr_to_var_vector[rtr] = rtr_var
        
    optimization_variables = {}
    tses = sorted(flow_values.keys())
    for ind, ts in enumerate(tses):
        if ts not in optimization_variables:
            optimization_variables[ts] = {}
        for rtr in router_set:
            optimization_variables[ts][rtr] = rtr_to_var_vector[rtr][ind]

    return optimization_variables, rtr_to_var_vector

def initialize_opti_vars(flow_values, router_set):
    optimization_variables = {}
    delta_vars = {}
    for ts in flow_values:
        if ts not in optimization_variables:
            optimization_variables[ts] = {}
            delta_vars[ts] = {}
        for rtr in router_set:
            optimization_variables[ts][rtr] = Variable()
            delta_vars[ts][rtr] = Bool()

    z_vars = {}
    for rtr in router_set:
        z_vars[rtr] = Variable()
    return optimization_variables, delta_vars, z_vars

def get_routers_from_assignments(flows_per_ts):
    rtr_set = set()
    for ts in flows_per_ts:
        rtr_set.update(flows_per_ts[ts].keys())
    return list(rtr_set)

def from_ts_to_rtr_key(ts_key, router_set):
    rtr_key = {}
    num_tses = len(ts_key)
    for ts in ts_key:
        for rtr in router_set:
            if rtr not in rtr_key:
                rtr_key[rtr] = []
            if rtr in ts_key[ts]:
                rtr_key[rtr].append(ts_key[ts][rtr])
            else:
               rtr_key[rtr].append(0)     
    return rtr_key

def get_flow_assignments(optimization_variables):
    optimal_assignments = {}
    for ts in optimization_variables:
        if ts not in optimal_assignments:
            optimal_assignments[ts] = {}
        for rtr in optimization_variables[ts]:
            optimal_assignments[ts][rtr] = optimization_variables[ts][rtr].value
    return optimal_assignments


def get_positivity_constraints(optimization_variables, z_vars):
    constraints = []
    for op_key_ts in optimization_variables:
        for op_key_rtr in optimization_variables[op_key_ts]:
            op_var = optimization_variables[op_key_ts][op_key_rtr]
            constraints.append(op_var >= 0)
    for rtr in z_vars:
        var = z_vars[rtr]
        constraints.append(var >= 0)
    return constraints

def get_peer_capacity_constraints(optimization_variables, ifspeeds):
    constraints = []
    for op_key_ts in optimization_variables:
        for op_key_rtr in optimization_variables[op_key_ts]:
            op_var = optimization_variables[op_key_ts][op_key_rtr]
            intf_capacity = ifspeeds[op_key_rtr]
            constraints.append(op_var <=  intf_capacity)
    return constraints

def get_demand_completion_constraints(optimization_variables, flow_per_ts):
    constraints = []
    for op_key_ts in optimization_variables:
        all_flows_at_ts = optimization_variables[op_key_ts].values()
        constraints.append(sum(all_flows_at_ts) >= flow_per_ts[op_key_ts])
    return constraints

def get_delta_constraints(delta_vars, router_set, k):
    delta_vars_by_rtr = from_ts_to_rtr_key(delta_vars, router_set)
    constraints = []
    for rtr in delta_vars_by_rtr:
        constraints.append(sum(delta_vars_by_rtr[rtr]) == k - 1)
    return constraints

def get_z_constraints(optimization_variables, delta_vars, z_vars, M):
    constraints = []
    optimization_variables_by_rtr = {}
    for ts in optimization_variables:
        for rtr in optimization_variables[ts]:
            if rtr not in optimization_variables_by_rtr:
                optimization_variables_by_rtr[rtr] = {ts: optimization_variables[ts][rtr]}
            else:
                optimization_variables_by_rtr[rtr][ts] = optimization_variables[ts][rtr]
                
    for rtr in z_vars:
        z = z_vars[rtr]
        for ts in optimization_variables_by_rtr[rtr]:
            rtr_assignment_in_ts = optimization_variables_by_rtr[rtr][ts]
            constraints.append(z >= (rtr_assignment_in_ts - delta_vars[ts][rtr]*M))
    return constraints
        
def get_constraints(optimization_variables, flow_per_ts, delta_vars,
                    z_vars, k, ifspeeds, router_set, M):
    all_constraints = []
    print "Get positivity constraints"
    positivity_constraints = get_positivity_constraints(optimization_variables, z_vars)
    all_constraints.extend(positivity_constraints)
    print "Total %d positivity constraints" % len(positivity_constraints)
    
    print "Get delta constraints"
    delta_constraints = get_delta_constraints(delta_vars, router_set, k)
    all_constraints.extend(delta_constraints)
    print "Total %d delta constraints" % len(delta_constraints)
    
    print "Get peer capacity constraints"
    peer_capacity_constraints = get_peer_capacity_constraints(optimization_variables,
                                                              ifspeeds)
    all_constraints.extend(peer_capacity_constraints)
    print "Total %d peer capacity constraints" % len(peer_capacity_constraints)
    
    print "Demand completion constraints"
    demand_completion_constraints = get_demand_completion_constraints(
        optimization_variables, flow_per_ts)
    all_constraints.extend(demand_completion_constraints)
    print "Total %d demand completion constraints" % len(demand_completion_constraints)
    
    print "Z constraints"
    z_constraints = get_z_constraints(optimization_variables, delta_vars, z_vars, M)
    all_constraints.extend(z_constraints)
    print "Total %d z constraints" % len(z_constraints)

    print "Total constraints:", len(all_constraints)
    return all_constraints

def solve_optimization_previous_work(flow_per_ts, demand_per_ts, router_set, total_cost,
                                     time_window, cluster_no, num_billing_slots, k_val,
                                     ifspeeds, m_val, combined_flow_file_name, **kwargs):
    strategy = kwargs.get('strategy', 'fixed-m')
    optimization_variables, rtr_to_var_vector = \
                                initialize_opti_vars_previous_work(flow_per_ts, router_set)
    all_constraints = []
    
    print "Get positivity constraints"
    for op_key_ts in optimization_variables:
        for op_key_rtr in optimization_variables[op_key_ts]:
            op_var = optimization_variables[op_key_ts][op_key_rtr]
            all_constraints.append(op_var >= 0)
            
    print "Get peer capacity constraints"
    peer_capacity_constraints = get_peer_capacity_constraints(optimization_variables,
                                                              ifspeeds)
    all_constraints.extend(peer_capacity_constraints)
    
    print "Demand completion constraints"
    demand_completion_constraints = get_demand_completion_constraints(
        optimization_variables, demand_per_ts)
    all_constraints.extend(demand_completion_constraints)

    num_slots = round(len(flow_per_ts)*0.1)
    print "Top 10% slots:", num_slots
    bw_cost_egress = 0
    for rtr in rtr_to_var_vector:
        bw_cost_egress += sum_largest(rtr_to_var_vector[rtr], num_slots)/num_slots
    
    obj = Minimize(bw_cost_egress)
    prob = Problem(obj, all_constraints)
    start_time = time.time()
    print "Solving.."
    try:
        prob.solve(solver=GUROBI, verbose=True)
        runtime = prob.solver_stats.solve_time
    except error.SolverError as e:
        print e
        print "Gurobi failed for time window", time_window
        return 

    end_time = time.time()
    print "Finished in %f seconds" % (end_time - start_time)
    print "status:", prob.status
    print "optimal value", prob.value
    optimal_assignments = get_flow_assignments(optimization_variables)
    per_router_optimized_assignments = from_ts_to_rtr_key(optimal_assignments, router_set)
    cost_per_rtr_optimal = calculate_cost_of_traffic_assignment(
        per_router_optimized_assignments, ifspeeds, num_billing_slots, k_val)
    total_cost_op = sum(cost_per_rtr_optimal.values())
    percent_saving = (total_cost - total_cost_op)*100/float(total_cost)
    print "Optimal cost:", total_cost_op, "Pre-op cost", total_cost
    print "Percent saving (%):", percent_saving
    print "Per router optimized_cost is", cost_per_rtr_optimal
    log_info_solver(combined_flow_file_name, time_window, total_cost,
                    total_cost_op, percent_saving, 15,
                    runtime, m_val, strategy)
    save_pre_post_opt_assignments_cluster(time_window, flow_per_ts, router_set,
                                          optimal_assignments, cluster_no,
                                          combined_flow_file_name, k_val, m_val,
                                          strategy)

    
def solve_optimization(flows_per_ts, demand_per_ts, router_set,
                       total_cost, time_window, cluster_no,
                       num_billing_slots, k_val, ifspeeds, m_val, combined_flow_file_name,
                       **kwargs):
    strategy = kwargs.get('strategy', 'fixed-m')
    if 'warm' in strategy:
        assert False, "Non-native optimization implementation does not support warm-start"
        
    print "Getting ready to solve the optimization for time window", time_window
    k = Parameter(sign="positive")
    k.value = round(k_val*num_billing_slots/100)
    M = Parameter(sign="positive")
    M.value = m_val

    print "Get optimization variables"
    optimization_variables, delta_vars, z_vars = \
                        initialize_opti_vars(flows_per_ts, router_set)
    print "Get constraints"
    all_constraints = get_constraints(optimization_variables, demand_per_ts,
                                      delta_vars, z_vars, k, ifspeeds, router_set, M)

    print "Get bandwidth cost"
    bw_cost_egress = calculate_peer_bw_cost(z_vars)
    all_constraints.append(bw_cost_egress <= total_cost)
    obj = Minimize(bw_cost_egress)
    prob = Problem(obj, all_constraints)
    start_time = time.time()
    mipgap = 0.10
    itlimit = 500000
    root_method = 3
    print "Solving.."
    try:
        print "Solver properties:"
        print "MIPGap:", mipgap
        print "root relaxation method:", root_method
        prob.solve(solver=GUROBI, verbose=True, PrePasses=3, MIPGap=mipgap,
                   MIPFocus=1,
                   #NodeMethod=0,
                   Heuristics=0.4,
                   ImproveStartTime=100,
                   ImproveStartGap=0.3,
                   TimeLimit=100000,
                   Method=root_method, Cuts=3)
        runtime = prob.solver_stats.solve_time
    except error.SolverError as e:
        print e
        print "Gurobi failed for time window", time_window
        return 

    end_time = time.time()
    print "Finished in %f seconds" % (end_time - start_time)
    print "status:", prob.status
    print "optimal value", prob.value
    
    optimal_assignments = get_flow_assignments(optimization_variables)
    per_router_optimized_assignments = from_ts_to_rtr_key(optimal_assignments, router_set)
    cost_per_rtr_optimal = calculate_cost_of_traffic_assignment(
        per_router_optimized_assignments, ifspeeds, num_billing_slots, k_val)
    total_cost_op = sum(cost_per_rtr_optimal.values())
    percent_saving = (total_cost - total_cost_op)*100/float(total_cost)
    print "Optimal cost:", total_cost_op, "Pre-op cost", total_cost
    print "Percent saving:", percent_saving
    print "Per router optimized_cost is", cost_per_rtr_optimal
    log_info_solver(combined_flow_file_name, time_window, total_cost,
                    total_cost_op, percent_saving, 15,
                    runtime, m_val, strategy)
    save_pre_post_opt_assignments_cluster(time_window, flows_per_ts, router_set,
                                          optimal_assignments, cluster_no,
                                          combined_flow_file_name, k_val, M.value, strategy)
    
def log_info_solver(combined_flow_file_name, time_window, total_cost,
                    total_cost_op, percent_saving,
                    mipgap, runtime, m_val, strategy):
    with open(SOLVER_LOG_FILE, "a") as fi:
        fi.write("%s,%s,%s,%d,%f,%f,%f,%f,%f\n" % (combined_flow_file_name, time_window, strategy,
                                                   m_val, total_cost, total_cost_op, percent_saving,
                                                   mipgap, runtime))
        
def calculate_peer_bw_cost(z_vars):
    cost = 0
    for rtr in z_vars:
        rtr_cost = cost_by_rtr(rtr)
        # print rtr_cost, rtr
        cost += rtr_cost * z_vars[rtr]
    return cost
def calculate_cost_of_traffic_assignment(rtr_to_ts_assignment, ifspeeds,
                                         num_billing_slots, k):
    per_rtr_cost = {}
    for rtr in rtr_to_ts_assignment:
        traffic_on_rtr_for_billing_period = rtr_to_ts_assignment[rtr]
        if len(traffic_on_rtr_for_billing_period) != num_billing_slots:
            diff = num_billing_slots - len(traffic_on_rtr_for_billing_period)
            assert diff > 0
            print "Completing empty slots", diff, rtr
            traffic_on_rtr_for_billing_period += [0] * diff
            
        rtr_capacity = ifspeeds[rtr]
        rtr_cost = cost_by_rtr(rtr)
        for per_ts_util in traffic_on_rtr_for_billing_period:
            try:
                assert round(per_ts_util) <= round(rtr_capacity)
            except:
                print rtr, per_ts_util, ifspeeds[rtr]

        per_rtr_cost[rtr] = rtr_cost * np.percentile(traffic_on_rtr_for_billing_period, 100-k)

    return per_rtr_cost

def save_pre_post_opt_assignments(month, non_opti_flows, opti_flows):
    csv_lines = [["ts", "rtr", "flow", "type"]]
    for ts in opti_flows:
        for rtr in opti_flows[ts]:
            csv_lines.append([ts, rtr, opti_flows[ts][rtr], "post-optimization"])
            if rtr in non_opti_flows:
                csv_lines.append([ts, rtr, non_opti_flows[ts][rtr], "pre-optimization"])
            else:
                csv_lines.append([ts, rtr, 0, "pre-optimization"])
            
    with open(TRAFFIC_ALLOCATIONS_GLOBAL_NW + "%s/%s_%d_%d.csv" %
              (peer_friendly_name, month, k_val, m_val), "w") as fi:
        writer = csv.writer(fi)
        writer.writerows(csv_lines)

def sanity_check(flows_ts_rtr, ifspeeds, peer_billed=None):
    for ts in flows_ts_rtr:
        for rtr in flows_ts_rtr[ts]:
            if peer_billed:
                rtr_key = '%s-%s' % (peer_billed, rtr)
            else:
                rtr_key = rtr
            if rtr_key in ifspeeds:
                ifsp = ifspeeds[rtr_key]
                vol = flows_ts_rtr[ts][rtr]
                if not vol <= ifsp:
                    pdb.set_trace()
            else:
                print "Didnt find capacity for", rtr
                continue

def save_pre_post_opt_assignments_cluster(time_window, non_opti_flows, router_set,
                                          opti_flows, cluster_no,
                                          combined_flow_file_name, k_val, m_val, strategy):
    csv_lines = [["ts", "rtr", "flow", "type"]]
    for ts in non_opti_flows:
        for rtr in router_set:
            if rtr not in non_opti_flows[ts]:
                csv_lines.append([ts, rtr, 0, "pre-optimization"])
                print "no flow pre opti", rtr
            else:
                csv_lines.append([ts, rtr, non_opti_flows[ts][rtr], "pre-optimization"])
            csv_lines.append([ts, rtr, opti_flows[ts][rtr], "post-optimization"])
            
    with open(TRAFFIC_ALLOCATIONS_CLUSTER_NW + "%s/%s_%d_%d_cluster%d_%s.csv" %
            (combined_flow_file_name, time_window, k_val, m_val, cluster_no, strategy), "w") as fi:
        writer = csv.writer(fi)
        writer.writerows(csv_lines)

def write_current_allocations(combined_flow_file_name,
                              input_clusters=None, num_clusters=None, overall=False):
    print "Churning.."
    df = pd.read_csv(CURRENT_TRAFFIC_ALLOCATIONS + "%s.csv" % combined_flow_file_name)
    clusterwise_billed_peer_flow_volumes_egress = {}
    router_set_by_cluster = {}
    for index, row in df.iterrows():
        if row[0] == 'ts': continue
        ts = row.ts
        rtr = row.rtr
        if overall:
            cluster = 9
        elif input_clusters:
            if rtr in input_clusters:
                cluster = input_clusters[rtr]
            else:
                print rtr
                assert False, "RTR to cluster mapping not found", rtr
        else:
            cluster = row.cluster

        peer = row.peer
        router_name = "%s-%s" % (peer, rtr)
        if cluster not in router_set_by_cluster:
            router_set_by_cluster[cluster] = []
        if router_name not in router_set_by_cluster[cluster]:
            router_set_by_cluster[cluster].append(router_name)
        gb = row.flow
        ts_dt = datetime.utcfromtimestamp(ts)
        month = "%d-%s" % (ts_dt.month, ts_dt.year)
        weeknumber = ts_dt.isocalendar()[1]
        week_num = "%d-%s" % (weeknumber, ts_dt.year)
        if month not in clusterwise_billed_peer_flow_volumes_egress:
            clusterwise_billed_peer_flow_volumes_egress[month] = {}
        if cluster not in clusterwise_billed_peer_flow_volumes_egress[month]:
            clusterwise_billed_peer_flow_volumes_egress[month][cluster] = {}

        if ts not in clusterwise_billed_peer_flow_volumes_egress[month][cluster]:
            clusterwise_billed_peer_flow_volumes_egress[month][cluster][ts] = {}
        if router_name not in clusterwise_billed_peer_flow_volumes_egress[month][cluster][ts]:
            clusterwise_billed_peer_flow_volumes_egress[month][cluster][ts][router_name] = gb 
        else:
            clusterwise_billed_peer_flow_volumes_egress[month][cluster][ts][router_name] += gb 


    for month in clusterwise_billed_peer_flow_volumes_egress:
        for cluster in clusterwise_billed_peer_flow_volumes_egress[month]:
            print "month, cluster", month, cluster
            pdb.set_trace()
            if num_clusters:
                fname = MONTHLY_CURRENT_TRAFFIC_ALLOCATIONS + \
                        "%s/num_clusters_%d/traffic_allocations_%s_cluster%s.json" % \
                        (combined_flow_file_name, num_clusters, month, cluster)
            else:
                fname = MONTHLY_CURRENT_TRAFFIC_ALLOCATIONS + \
                        "%s/traffic_allocations_%s_cluster%s.json" % \
                        (combined_flow_file_name, month, cluster)
            print fname
            with open(fname, "w") as fi:
                json.dump(clusterwise_billed_peer_flow_volumes_egress[month][cluster], fi)

def read_current_allocations(combined_flow_file_name, month, cluster, **kwargs):
    num_clusters = kwargs.get('num_clusters', None)
    if num_clusters:
        fname = MONTHLY_CURRENT_TRAFFIC_ALLOCATIONS + \
                "%s/num_clusters_%d/traffic_allocations_%s_cluster%s.json" % \
                (combined_flow_file_name, num_clusters, month, cluster)
    else:
        fname = MONTHLY_CURRENT_TRAFFIC_ALLOCATIONS + \
                "%s/traffic_allocations_%s_cluster%s.json" % \
                (combined_flow_file_name, month, cluster)
    data = None
    if os.path.isfile(fname):
        with open(fname) as fi:
            data = json.load(fi)
    else:
        print "File name not found", fname
    return data

def extract_weekly_allocations(monthly_flow_allocations):
    weekly_allocations = {}
    for ts in monthly_flow_allocations:
        ts_dt = datetime.utcfromtimestamp(int(ts))
        weeknumber = ts_dt.isocalendar()[1]
        week_num = "%d-%s" % (weeknumber, ts_dt.year)
        if week_num not in weekly_allocations:
            weekly_allocations[week_num] = {}

        weekly_allocations[week_num][ts] = monthly_flow_allocations[ts]
    return weekly_allocations
