## Example command to run this program:
from sklearn.cluster import KMeans
from vincenty import vincenty
import time
import numpy as np
from datetime import datetime
import csv
import pdb
from cvxpy import *
import sys
import numpy as np
import pandas as pd
from consts import *

if sys.argv[4] == 'native':
    print "Using native Gurobi bindings for the solver"
    from api_opt_native import *
elif sys.argv[4] == 'non_native':
    print "Using CVX wrapper for Gurobi"
    from api_opt import *

k_val = int(sys.argv[1]) # percentile (1-99)
m_val = int(sys.argv[2])
combined_flow_file_name = sys.argv[3]

if len(sys.argv) > 5:
    model_strategy = sys.argv[5]
else:
    model_strategy = "fixed-m"

if 'cluster' in model_strategy:
    num_link_clusters = int(model_strategy.split('cluster-')[-1])
        
print "K value:", k_val
print "M value:", m_val
print "Model Strategy:", model_strategy

with open(EDGE_INTF_CAPACITY_TIME_FNAME) as fi:
    ifspeeds_pre = json.load(fi)

ifspeeds = {}
for rtr in ifspeeds_pre:
    for peer in ifspeeds_pre[rtr]:
        ifspeeds["%s-%s" % (peer.upper(), rtr)] = max(ifspeeds_pre[rtr][peer].values()) *5*60

def get_subset_of_flows(cluster_flows):
    cluster_sub = {}
    for ts in cluster_flows:
        cluster_sub[ts] = {}
        for rtr in cluster_flows[ts]:
            if rtr not in links_to_include:
                continue
            cluster_sub[ts][rtr] = cluster_flows[ts][rtr]

    return cluster_sub

active_links_per_month = {}
previous_model = None
for month in [
        '6-2018',
        '7-2018',
        '8-2018',
        '9-2018',
        '10-2018',
        '11-2018',
        '12-2018',
        '1-2019',
        '2-2019',
        '3-2019',
        '4-2019',
        '5-2019',
        '6-2019'
        "7-2019",
        "8-2019",
        "9-2019",
        "10-2019",
        "11-2019",
        "12-2019",
        "1-2020",
        "2-2020",
        "3-2020",
        "4-2020",
        "5-2020"
]:
    print "Month:", month
    #for cluster_id in [ 0, 1, 2, 3, 4]:
    for cluster_id in [9]:
        cluster_sub_flows = read_current_allocations(combined_flow_file_name,
                                                     month, cluster_id, strategy=model_strategy)
        if not cluster_sub_flows: continue
        
        if model_strategy == 'subset':
            cluster_sub_flows = get_subset_of_flows(cluster_sub_flows)
            
        router_set = get_routers_from_assignments(cluster_sub_flows)
        if not router_set:
            print "No router links included in analysis, skipping"
            continue
        
        print router_set
        num_billing_slots = len(cluster_sub_flows)
        print 'Calculating optimum traffic assignments for month %s with %d billing slots' %\
            (month, num_billing_slots)
        sanity_check(cluster_sub_flows, ifspeeds)
        # Calculate the present-day traffic distribution per peering link
        per_router_present_dist = from_ts_to_rtr_key(cluster_sub_flows, router_set)
        cost_per_rtr = calculate_cost_of_traffic_assignment(per_router_present_dist,
                                                            ifspeeds, num_billing_slots, k_val)

        print "Minimixing the %d-ile element" % k_val
        
        # Calculate demand per 5 minute interval
        flows_per_ts_egress = {}
        for ts in cluster_sub_flows:

            flows_per_ts_egress[ts] = sum(cluster_sub_flows[ts].values())

        total_cost = sum(cost_per_rtr.values())
        print "Pre-optimization cost is:", total_cost
        print "Per router pre-op cost is", cost_per_rtr

        if model_strategy == 'prev_work':
            model = solve_optimization_previous_work(cluster_sub_flows, flows_per_ts_egress,
                                        router_set,
                                        total_cost, month, cluster_id, num_billing_slots,
                                        k_val, ifspeeds, m_val, combined_flow_file_name,
                                        previous_model=previous_model, strategy=model_strategy)

        else:
            model = solve_optimization(cluster_sub_flows, flows_per_ts_egress,
                                        router_set,
                                        total_cost, month, cluster_id, num_billing_slots,
                                        k_val, ifspeeds, m_val, combined_flow_file_name,
                                        previous_model=previous_model, strategy=model_strategy,
                                        scav_frac = scav_frac)

        print "Finished saving to disk", cluster_id, month
        previous_model = model
