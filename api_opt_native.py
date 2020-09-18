import radix
import calendar
from gurobipy import *
import time
import os
import json
from datetime import datetime
import pandas as pd
from consts import * # This has all the data file paths
import pdb
import csv
import numpy as np

def minimize_changes_in_allocation(optimization_variables, model, ifspeeds,
                                   num_billing_slots):
    tses_sorted = sorted(optimization_variables.keys())
    previous_allocations = {}
    sum_abs = {}
    for ts in tses_sorted:
        current_allocations = {}        
        for rtr in optimization_variables[ts]:
            current_allocations[rtr] = optimization_variables[ts][rtr]
        for rtr in current_allocations:
            if rtr not in sum_abs:
                sum_abs[rtr] = 0
            if rtr in previous_allocations:
                rtr_cost = cost_by_rtr(rtr)
                rtr_capacity = ifspeeds[rtr]
                y = model.addVar(lb=-rtr_capacity,
                                 ub=rtr_capacity, name="y_%s_%s" % (str(ts), rtr))
                abs_y = model.addVar(lb=0,
                            ub=rtr_capacity, name = "abs_y_%s_%s" % (str(ts), rtr))
                model.addConstr(y, GRB.EQUAL,
                                current_allocations[rtr] - previous_allocations[rtr])
                model.addGenConstrAbs(abs_y, y)
                sum_abs[rtr] += abs_y * rtr_cost
                
        previous_allocations = current_allocations

    for rtr in sum_abs:
        sum_abs[rtr] = sum_abs[rtr]/num_billing_slots

    return model, sum(sum_abs.values())

def minimize_changes_in_allocation_fixed(optimization_variables, model,
                                         ifspeeds, fraction_cap,
                                         num_billing_slots):
    sum_abs = 0
    for ts in optimization_variables:
        for rtr in optimization_variables[ts]:
            rtr_cost = cost_by_rtr(rtr)
            rtr_capacity = ifspeeds[rtr]
            y = model.addVar(lb=-rtr_capacity, ub=rtr_capacity, name="y_%s_%s" % (str(ts), rtr))
            abs_y = model.addVar(lb=0, ub=rtr_capacity, name = "abs_y_%s_%s" % (str(ts), rtr))
            model.addConstr(y, GRB.EQUAL,
                            optimization_variables[ts][rtr] - fraction_cap*rtr_capacity)
            model.addGenConstrAbs(abs_y, y)
            sum_abs += abs_y * rtr_capacity

    return model, sum_abs
            
def initialize_opti_vars(flow_values, router_set, model, ifspeeds, bound_by_cap = False,
                         scavenger=100):
    optimization_variables = {}
    delta_vars = {}
    for ts in flow_values:
        ts_dt = datetime.utcfromtimestamp(int(ts))
        hour = ts_dt.hour
        minute = ts_dt.minute
        day = ts_dt.day
        ts_str = "%d_%d_%d" % (hour, minute, day)
        if ts not in optimization_variables:
            optimization_variables[ts] = {}
            delta_vars[ts] = {}
        for rtr in router_set:
            if rtr in flow_values[ts]:
                lb = (100 - scavenger) * flow_values[ts][rtr]/100.0
            else:
                lb = 0
            optimization_variables[ts][rtr] = model.addVar(lb=lb, name="x_%s_%s" % (ts_str, rtr))
            delta_vars[ts][rtr] = model.addVar(vtype=GRB.BINARY, name="delta_%s_%s" % (ts_str, rtr))

    z_vars = {}
    for rtr in router_set:
        if bound_by_cap:
            lb_by_capacity_fraction = ifspeeds[rtr] * 0.1
        else:
            lb_by_capacity_fraction = 0
        z_vars[rtr] = model.addVar(lb=lb_by_capacity_fraction, name="z_%s" % rtr)
    return optimization_variables, delta_vars, z_vars, model

def assign_fraction_cap_start_variables(flow_values, router_set, new_model, ifspeeds,
                                        capacity_fraction):
    for ts in flow_values:
        total_demand = sum(flow_values[ts].values())
        ts_dt = datetime.utcfromtimestamp(int(ts))
        hour = ts_dt.hour
        minute = ts_dt.minute
        day = ts_dt.day
        ts_str = "%d_%d_%d" % (hour, minute, day)

        for rtr in router_set:
            rtr_capacity = ifspeeds[rtr]
            var_new = new_model.getVarByName("x_%s_%s" % (ts_str, rtr))
            var_new.start = rtr_capacity * capacity_fraction

    return new_model

def assign_warm_start_variables(flow_values, router_set, new_model, old_model):
    for rtr in router_set:
        var_old = old_model.getVarByName("z_%s" % rtr)
        var_new = new_model.getVarByName("z_%s" % rtr)
        if var_new and var_old:
            var_new.start = var_old.x
            
    for ts in flow_values:
        ts_dt = datetime.utcfromtimestamp(int(ts))
        hour = ts_dt.hour
        minute = ts_dt.minute
        day = ts_dt.day
        ts_str = "%d_%d_%d" % (hour, minute, day)
        for rtr in router_set:
            var_old = old_model.getVarByName("x_%s_%s" % (ts_str, rtr))
            var_new = new_model.getVarByName("x_%s_%s" % (ts_str, rtr))
            if var_old and var_new:
                var_new.start = var_old.x

    return new_model
            
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
            optimal_assignments[ts][rtr] = optimization_variables[ts][rtr].x
    return optimal_assignments


def get_positivity_constraints(optimization_variables, z_vars, model):
    constraints = []
    for op_key_ts in optimization_variables:
        for op_key_rtr in optimization_variables[op_key_ts]:
            op_var = optimization_variables[op_key_ts][op_key_rtr]
            constraints.append(op_var >= 0)
    for rtr in z_vars:
        var = z_vars[rtr]
        constraints.append(var >= 0)
    return constraints, model

def get_peer_capacity_constraints(optimization_variables, ifspeeds, model):
    for op_key_ts in optimization_variables:
        for op_key_rtr in optimization_variables[op_key_ts]:
            op_var = optimization_variables[op_key_ts][op_key_rtr]
            intf_capacity = ifspeeds[op_key_rtr]
            model.addConstr(op_var <=  intf_capacity)
    return model

def get_demand_completion_constraints(optimization_variables, flow_per_ts, model):
    for op_key_ts in optimization_variables:
        all_flows_at_ts = optimization_variables[op_key_ts].values()
        model.addConstr(sum(all_flows_at_ts) >= flow_per_ts[op_key_ts])
    return model
    
def get_delta_constraints(delta_vars, router_set, k, model):
    delta_vars_by_rtr = from_ts_to_rtr_key(delta_vars, router_set)
    for rtr in delta_vars_by_rtr:
        model.addConstr(sum(delta_vars_by_rtr[rtr]) == k - 1)
    return model

def get_z_constraints(optimization_variables, delta_vars, z_vars, M, model,
                      ifspeeds, strategy="fixed-m"):
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
            if strategy == 'link-capacity':
                M = ifspeeds[rtr]
            else:
                assert M
            model.addConstr(z >= (rtr_assignment_in_ts - delta_vars[ts][rtr]*M))
    return model

def log_info_solver(combined_flow_file_name, cluster, time_window, total_cost,
                    total_cost_op, percent_saving,
                    mipgap, runtime, m_val, strategy):
    with open(SOLVER_LOG_FILE, "a") as fi:
        fi.write("%s,%s,%s,%d,%f,%f,%f,%f,%f\n" % (combined_flow_file_name, time_window,
                                                   strategy + "-%d" % cluster,
                                                   m_val, total_cost, total_cost_op, percent_saving,
                                                   mipgap, runtime))
        
def get_constraints(optimization_variables, flow_per_ts, delta_vars,
                    z_vars, k, ifspeeds, router_set, m_val, model, strategy="fixed-m"):
    all_constraints = []
    print "Get delta constraints"
    model = get_delta_constraints(delta_vars, router_set, k, model)
    
    print "Get peer capacity constraints"
    model = get_peer_capacity_constraints(optimization_variables, ifspeeds, model)
    
    model = get_z_constraints(optimization_variables, delta_vars, z_vars, m_val,
                              model, ifspeeds, strategy)

    return model

def solve_optimization(flows_per_ts, demand_per_ts, router_set,
                       total_cost, time_window, cluster_no,
                       num_billing_slots, k_val, ifspeeds, m_val, combined_flow_file_name,
                       **kwargs):
    '''
    flows_per_ts has the format {ts1: {rtr1: flow1, rtr2: flow2 ..}, ts2: ..}
    '''
    previous_model = kwargs.get('previous_model', None)
    strategy = kwargs.get('strategy', 'fixed-m')
    if 'bound' in strategy:
        print "Link costs bounded by 10% of link capacities"
        bound_by_cap = True
    else:
        bound_by_cap = False

    if 'scavenger' in strategy:
        scav_frac = kwargs.get('scav_frac', 40)
        print "Scavenger traffic volume:", scav_frac
    else:
        scav_frac = 100 

    timelimit = kwargs.get("timelimit", 18000)
    print "Getting ready to solve the optimization for time window", time_window
    model = Model("mip")
    k = round(k_val*num_billing_slots/100)
    
    print "Get optimization variables"
    optimization_variables, delta_vars, z_vars, model = \
                        initialize_opti_vars(flows_per_ts, router_set, model, ifspeeds,
                                             bound_by_cap=bound_by_cap, scavenger = scav_frac)
    model.update()
    print "Get constraints"
    model = get_constraints(optimization_variables, demand_per_ts,
                            delta_vars, z_vars, k, ifspeeds,
                            router_set, m_val, model, strategy)
    print "Get bandwidth cost"
    bw_cost_egress = calculate_peer_bw_cost(z_vars)    
    # model, change_in_allocs = \
    #                minimize_changes_in_allocation(optimization_variables, model, ifspeeds,
    #                                               num_billing_slots)

    # model, change_in_allocs = \
    #             minimize_changes_in_allocation_fixed(optimization_variables, model, ifspeeds, 0.1,
    #                                                  num_billing_slots)

    # epsilon = 0.1
    # print "Epsilon:", epsilon
    # objective = bw_cost_egress + (epsilon * change_in_allocs)
    
    objective = bw_cost_egress
    model.addConstr(bw_cost_egress <= total_cost)
    model.setObjective(objective)
    
    model.write(MODEL_SAVE + "%s_%s.lp" % (time_window, strategy))
    start_time = time.time()
    model.setParam("mipgap", 0.15)
    # model.setParam("GomoryPasses", 15)
    model.setParam("IterationLimit", 5000000)
    model.setParam("heuristics", 0.5)
    model.setParam("timelimit", timelimit)
    model.setParam("improvestarttime", 100)
    model.setParam("improvestartgap", 0.3)
    #model.setParam("Sifting", 2)
    model.setParam("Method", 3)
    #model.setParam("Crossover", 4)

    # Warm start from previous allocations
    if "warm" in strategy and previous_model:
        print "Assigning variable values from previous optimal assignment: warm start"
        model = assign_warm_start_variables(flows_per_ts, router_set, model, previous_model)
    elif "frac_cap" in strategy:
        print "Setting init allocations to be fraction of link capacities"
        model = assign_fraction_cap_start_variables(flows_per_ts, router_set, model, ifspeeds,
                                                    0.2)
    model.update()    
    print "Solving.."
    try:
        print "Solver properties:"
        model.optimize()
        runtime = model.Runtime
        mipgap = model.MIPGap * 100 # Percent further from the LP optimal
    except error.SolverError as e:
        print e
        print "Gurobi failed for time window", time_window
        return 

    end_time = time.time()

    print "Finished in %f seconds" % (end_time - start_time)

    if model.SolCount == 0:
        print "Solver failed, no solution."
        return None
    
    optimal_assignments = get_flow_assignments(optimization_variables)
    per_router_optimized_assignments = from_ts_to_rtr_key(optimal_assignments, router_set)
    cost_per_rtr_optimal = calculate_cost_of_traffic_assignment(
        per_router_optimized_assignments, ifspeeds, num_billing_slots, k_val)
    total_cost_op = sum(cost_per_rtr_optimal.values())
    percent_saving = (total_cost - total_cost_op)*100/float(total_cost)
    print "Optimal cost:", total_cost_op, "Pre-op cost", total_cost
    print "Percent saving:", percent_saving
    print "Per router optimized_cost is", cost_per_rtr_optimal
    log_info_solver(combined_flow_file_name,cluster_no, time_window, total_cost,
                    total_cost_op, percent_saving, mipgap,
                    runtime, m_val, strategy)
    save_pre_post_opt_assignments_cluster(time_window, flows_per_ts, router_set,
                                          optimal_assignments, cluster_no,
                                          combined_flow_file_name, k_val, m_val, strategy)
    return model

def calculate_peer_bw_cost(z_vars):
    cost = 0
    for rtr in z_vars:
        rtr_cost = cost_by_rtr(rtr)
        cost += rtr_cost * z_vars[rtr]
    return cost
        
def cost_by_rtr(rtr):
    print "Enter your peering rates here"


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
                pdb.set_trace()
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

def parse_raw_ipfix_to_combined_file(IPFIX_DUMP, combined_file_name, month_number_global=None,
                                     cluster_map=None):
    def get_cluster(link):
        if not cluster_map:
            return 9
        if link in cluster_map:
            return cluster_map[link]

    print "Enter your peer subset here"
    peer_subset = []
    fd = open(CURRENT_TRAFFIC_ALLOCATIONS +  combined_file_name + ".csv", "w")
    fd.write("ts,rtr,peer,cluster,flow\n")
    with open(IPFIX_DUMP) as fi:
        reader = csv.reader(fi)
        for row in reader:
            tsstr = row[0]
            try:
                dt_ts = datetime.strptime(tsstr, '%m/%d/%Y %I:%M:%S %p')
            except:
                pdb.set_trace()
            if month_number_global and dt_ts.month != month_number_global:
                continue
            ts_int = calendar.timegm(dt_ts.timetuple())
            rtr = row[1]
            peerlink = row[2]
            if peerlink not in peer_subset: continue
            direction = row[3]
            if direction != "OUTBOUND": continue
            cluster = get_cluster(peerlink)
            bytes = int(row[4])
            mb = bytes * 8/(1024*1024.0)
            fd.write("%s,%s,%s,%s,%f\n" % (ts_int, rtr, peerlink, cluster, mb))
    fd.close()
    
def parse_metro_pfx_mapping(fname):
    rtree = radix.Radix()
    pfx_to_metro = {}
    with open(fname) as fi:
        reader = csv.reader(fi)
        for row in reader:
            if row[0] == 'prefix': continue
            pfx = row[0]
            metro = row[1]
            try:
                rnode = rtree.add(pfx)
            except:
                pdb.set_trace()
            rnode.data["metro"] = metro
            if pfx not in pfx_to_metro:
                pfx_to_metro[pfx] = metro
            else:
                assert False, pfx_to_metro[pfx]
    return pfx_to_metro, rtree
                
def parse_raw_ipfix_to_combined_file_clientmetro(IPFIX_DUMP_CLIENT_PFX, combined_file_name,
                                                 month_number_global=None,
                                                 cluster_map=None):
    pfx_to_metro, rtree = parse_metro_pfx_mapping(METRO_PFX_MAP)
    rtr_to_peer = map_rtr_ip_to_peerlink()
    def get_metro(pfx):
        if pfx in pfx_to_metro:
            return [pfx_to_metro[pfx]]
        rnodes = rtree.search_covered(pfx)
        if not rnodes: return ["Unknown"]
        return list(set([x.data["metro"] for x in rnodes]))

    print "Enter your peer subset here"
    peer_subset = []
    nometro = set()
    peerlink_to_client_metro = {}
    with open(CURRENT_TRAFFIC_ALLOCATIONS + IPFIX_DUMP_CLIENT_PFX) as fi:
        reader = csv.reader(fi)
        for row in reader:
            tsstr = row[0]
            try:
                dt_ts = datetime.strptime(tsstr, '%m/%d/%Y %I:%M:%S %p')
            except:
                pdb.set_trace()
            if month_number_global and dt_ts.month != month_number_global:
                continue
            ts_int = calendar.timegm(dt_ts.timetuple())
            rtr = row[1]
            rtrip = row[2]
            ifindex = row[3]
            if (rtr, rtrip, ifindex) not in rtr_to_peer: continue
            peerasn = rtr_to_peer[(rtr, rtrip, ifindex)]
            asn = "AS%d" % peerasn
            asname = "ENTER AS MAPPINGS HERE"
            peerlink = "%s-%s" % (asn, asname)
            assert peerlink in peer_subset
            peerlink = "%s-%s" % (peerlink,  rtr)
            
            if peerlink not in peerlink_to_client_metro:
                peerlink_to_client_metro[peerlink] = {}

            if ts_int not in peerlink_to_client_metro[peerlink]:
                peerlink_to_client_metro[peerlink][ts_int] = {}
            
            dstpfx = row[4]
            dstasn = row[5]
            metrolist = get_metro(dstpfx)
            num_metros = len(metrolist)
            for metro in metrolist:
                client_metro_asn = "%s-%s" % (metro, dstasn)
                mb = float(row[6])
                mb = mb / 1024.0 # Getting it in Gbs
                mb = mb/num_metros
                if client_metro_asn not in peerlink_to_client_metro[peerlink][ts_int]:
                    peerlink_to_client_metro[peerlink][ts_int][client_metro_asn] = 0
                peerlink_to_client_metro[peerlink][ts_int][client_metro_asn] += mb

    with open(CURRENT_TRAFFIC_ALLOCATIONS + combined_file_name + ".json", "w") as fi:
        json.dump(peerlink_to_client_metro, fi)    

def client_to_pop_stickiness(combined_file_name):
    '''
    Based on IPFIX data, this function maps which clients (metro + ASN)
    were served content from which set of peer links (RTR + PEER).
    '''
    with open(CURRENT_TRAFFIC_ALLOCATIONS + combined_file_name + ".json") as fi:
        peerlink_to_client_metro = json.load(fi)

    client_to_links = {}
    for peerlink in peerlink_to_client_metro:
        for ts in peerlink_to_client_metro[peerlink]:
            for metroasn in peerlink_to_client_metro[peerlink][ts]:
                if metroasn not in client_to_links:
                    client_to_links[metroasn] = []
                if peerlink not in client_to_links[metroasn]:
                    client_to_links[metroasn].append(peerlink)

    return client_to_links

def write_current_allocations_by_cluster(combined_flow_file_name, final_dir,
                                         peer_to_cluster_map, overall=False):
    df = pd.read_csv(CURRENT_TRAFFIC_ALLOCATIONS + "%s.csv" % combined_flow_file_name)
    clusterwise_billed_peer_flow_volumes_egress = {}
    router_set_by_cluster = {}
    for index, row in df.iterrows():
        if row[0] == 'ts': continue
        ts = row.ts
        rtr = row.rtr
        peer = row.peer
        # peerasn = row.peer.split('-')[0].split('AS')[-1]
        peerlink = "%s-%s" % (rtr, peer)
        if overall:
            cluster = 9
        elif peer_to_cluster_map:
            if peerlink in peer_to_cluster_map:
                cluster = peer_to_cluster_map[peerlink]
            else:
                continue
        else:
            cluster = row.cluster

        peer = row.peer
        router_name = "%s-%s" % (peer, rtr)
        if cluster not in router_set_by_cluster:
            router_set_by_cluster[cluster] = []
        if router_name not in router_set_by_cluster[cluster]:
            router_set_by_cluster[cluster].append(router_name)
        gb = row.flow/1024.0
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
            with open(MONTHLY_CURRENT_TRAFFIC_ALLOCATIONS_PERF + \
                      "%s/traffic_allocations_%s_cluster%s.json" %
                      (final_dir, month, cluster), "w") as fi:
                json.dump(clusterwise_billed_peer_flow_volumes_egress[month][cluster], fi)

def write_current_allocations(combined_flow_file_name,
                              cluster_fname=None, overall=False):
    df = pd.read_csv(CURRENT_TRAFFIC_ALLOCATIONS + "%s.csv" % combined_flow_file_name)
    peer_to_cluster_map = None
        
    clusterwise_billed_peer_flow_volumes_egress = {}
    router_set_by_cluster = {}
    for index, row in df.iterrows():
        if row[0] == 'ts': continue
        ts = row.ts
        rtr = row.rtr
        peerasn = row.peer.split('-')[0].split('AS')[-1]
        peerlink = "%s-%s" % (rtr, peerasn)
        if overall:
            cluster = 9
        elif peer_to_cluster_map:
            if peerlink in peer_to_cluster_map:
                cluster = peer_to_cluster_map[peerlink]
            else:
                continue
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
            with open(MONTHLY_CURRENT_TRAFFIC_ALLOCATIONS_PERF + \
                      "%s/traffic_allocations_%s_cluster%s.json" %
                      (combined_flow_file_name, month, cluster), "w") as fi:
                json.dump(clusterwise_billed_peer_flow_volumes_egress[month][cluster], fi)

def map_rtr_ip_to_peerlink():
    rtr_to_peer = {}
    with open(CURRENT_TRAFFIC_ALLOCATIONS + RTR_IP_TO_PEER_MAP) as fi:
        reader = csv.reader(fi)
        for row in reader:
            if row[1] == "DeviceIp": continue
            rtr = row[0]
            ip = row[1]
            ifint = row[2]
            try:
                asn = int(row[3])
            except: pdb.set_trace()
            if (rtr, ip, ifint) in rtr_to_peer:
                pdb.set_trace()
            rtr_to_peer[(rtr, ip, ifint)] = asn                    
    return rtr_to_peer

def read_current_allocations(combined_flow_file_name, month, cluster, strategy, test=False):
    fname = MONTHLY_CURRENT_TRAFFIC_ALLOCATIONS_PERF + \
            "%s/traffic_allocations_%s_cluster%s.json" % \
            (combined_flow_file_name, month, cluster)
    data = None
    if os.path.isfile(fname):
        with open(fname) as fi:
            data = json.load(fi)
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
