import timeit, functools
from peer_to_links_map import *
import os
import sys
sys.path.append(os.path.abspath('..'))
from  api_opt_native import *
import random
import json
from heapq import *
import math
from datetime import datetime
import glob
import numpy as np
import pdb
import csv
import sys
from consts import *

combined_flow_file_name = None
def read_allocation(month, cluster_no):
    rtr_to_cluster = {}
    cluster_to_rtr = {}
    if cluster_no == 9:
        fnames = glob.glob(TRAFFIC_ALLOCATIONS_CLUSTER_NW +
                           "%s/%s_%d_%d_cluster%s_warm-fixed-m.csv" %
                           (combined_flow_file_name, month, 5, 10000, cluster_no))
    else:
        fnames = glob.glob(TRAFFIC_ALLOCATIONS_CLUSTER_NW + 
                           "%s/%s_%d_%d_cluster%s_warm-fixed-m-perf.csv" %
                           (combined_flow_file_name, month, 5, 10000, cluster_no))
    if not fnames: return
    fname = fnames[0]
    pre_op = {}
    post_op = {}
    pre_op_by_rtr = {}
    post_op_by_rtr = {}
    set_tses = set()        
    link_set = set()
    with open(fname) as fi:
        reader = csv.reader(fi)
        for row in reader:
            if row[0] == 'ts': continue
            ts = int(row[0])
            set_tses.add(ts)
            if ts not in pre_op:
                pre_op[ts] = {}
            if ts not in post_op:
                post_op[ts] = {}

            link = row[1]
                    
            if link not in pre_op_by_rtr:
                pre_op_by_rtr[link] = {}

            if link not in post_op_by_rtr:
                post_op_by_rtr[link] = {}
                    
            link_set.add(link)
            rtr_to_cluster[link] = cluster_no
            value = float(row[2])
            
            try:
                if row[-1] == 'pre-optimization':
                    assert ts not in pre_op_by_rtr[link]
                    assert link not in pre_op[ts]
                    pre_op_by_rtr[link][ts] = value
                    pre_op[ts][link] = value
                elif row[-1] == 'post-optimization':
                    assert link not in post_op[ts]
                    assert ts not in post_op_by_rtr[link]
                    post_op[ts][link] = value
                    post_op_by_rtr[link][ts] = value
            except AssertionError:
                pdb.set_trace()

        cluster_to_rtr[cluster_no] = list(link_set)
        print "Cluster number:", cluster_no, "has %d links" % len(link_set)
        
    post_op_by_rtr_list = {}
    pre_op_by_rtr_list = {}
    for rtr in post_op_by_rtr:
        post_op_by_rtr_list[rtr] = post_op_by_rtr[rtr].values()
    for rtr in pre_op_by_rtr:
        pre_op_by_rtr_list[rtr] = pre_op_by_rtr[rtr].values()
            
    return pre_op, post_op, pre_op_by_rtr_list, post_op_by_rtr_list, link_set

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

class PiorityQueue:
    def __init__(self, ifspeeds):
        self.entry_finder = {}
        self.linkheap = []
        self.ifspeeds = ifspeeds
        self.REMOVED = '<removed-task>'
        
    def get_bca(self, used_links):
        # BCA is burstable capacity at a given point of time.
        # It captures the value by which we can burst if all links
        # with a free slot burst
        bca = 0
        for link in self.entry_finder:
            if self.entry_finder[link][2] > 0:
                if link in used_links:
                    # If this link already has some traffic assigned to it
                    # in the present time slot, its capacity for bursting
                    # is reduced by the traffic it is already carrying.
                    bca += self.ifspeeds[link] - used_links[link]
                else:
                    bca += self.ifspeeds[link]
        return bca
    
    def add_link(self, linkname, slots, priority=0):
        if linkname in self.entry_finder:
            self.remove_link(linkname)
        capacity = self.ifspeeds[linkname]
        # entry = [priority, capacity, slots, linkname]
        entry = [priority, slots, linkname]
        self.entry_finder[linkname] = entry
        heappush(self.linkheap, entry)

    def remove_link(self,linkname):
        'Mark an existing link as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(linkname)
        entry[-1] = self.REMOVED

    def pop_link(self):
        'Remove and return the lowest priority link. Raise KeyError if empty.'
        while self.linkheap:
            # priority, capacity, slots, link = heappop(self.linkheap)
            priority, slots, link = heappop(self.linkheap)
            if link is not self.REMOVED and slots > 0:
                del self.entry_finder[link]
                return link, slots, priority
        raise KeyError('pop from an empty priority queue')

def sigcomm04_heuristic_offline(demand, links, ifspeeds):
    FREE_SLOTS = round(len(demand)*5/100.0)  - 1
    def select_random_links_with_slots(links, slots, p_k, k):
        link_subset = [link for link in links if slots[link] > 0]
        if len(link_subset) == 0:
            return []
        elif len(link_subset) >= k:
            return random.sample(link_subset, k)
        else:
            return link_subset
    
    def is_assignable(demands, V_0, p_k):
        slots = {}
        for link in links:
            slots[link] = FREE_SLOTS

        for ts in demands:
            demand = demands[ts]
            if demand <= V_0: continue
            demand_met = False
            for j in range(1, len(links)):
                peak_links = select_random_links_with_slots(links, slots, p_k, j)
                burst_cap = sum([ifspeeds[x] for x in peak_links])
                burst_cap_deduct = sum([p_k[x] for x in peak_links])
                burst_cap = V_0 + burst_cap - burst_cap_deduct
                if burst_cap >= demand:
                    demand_met = True
                    break
                
            if demand_met:
                for link in peak_links:
                    slots[link] -= 1
            else:
                return False
                    
        for link in slots:
            assert slots[link] >= 0
        
        return True
                
    sorted_tses = sorted(demand.keys())
    I = len(sorted_tses)
    delta = 0.01
    f = min([len(links) * 0.05, 1])
    sat = False
    while not sat:
        V_0 = np.percentile(demand.values(), (1-f)*100)
        # print V_0, (1-f)* 100
        link_costs = [(link, cost_by_rtr(link)) for link in links]
        link_costs = sorted(link_costs, key = lambda x: x[1])
        remaining = V_0
        p_k = {}
        for link, cost in link_costs:
            if remaining <= 0:
                p_k[link] = 0
                continue
            cap = ifspeeds[link]
            if remaining < cap:
                p_k[link] = remaining
                remaining = 0
            else:
                p_k[link] = cap
                remaining = remaining - cap
        
        sat = is_assignable(demand, V_0, p_k)
        f = f - delta
        if f < 0: return None, None, None

    if sat:
        cost = sum([p_k[link]*cost_by_rtr(link) for link in p_k])
        return p_k, cost, V_0
    else:
        return None, None, None

def bin_pack_links(links, ifspeeds, traffic_volume):
    ''' Given links with different costs and capacities,
    this method assigns billable bandwidth to the links such that
    the cost is minimzed and capacities are respected.
    '''
    links_by_cost = {}
    for link in links:
        link_cost = cost_by_rtr(link)
        if link_cost not in links_by_cost:
            links_by_cost[link_cost] = []
        links_by_cost[link_cost].append(link)
        
    remaining_volume = traffic_volume
    link_assignments = {}
    
    while remaining_volume > 0:
        # No links remain with available capacity
        if not links_by_cost:
            print "Cannot satisfy %f with this link set" % traffic_volume
            return None
        
        # Cheapest set of links
        link_set = sorted(links_by_cost.iteritems(), key=lambda x:x[0])[0][1]
        link_cost = sorted(links_by_cost.iteritems(), key=lambda x:x[0])[0][0]
        link_set_capacity = sum([ifspeeds[x] for x in link_set])
        
        if link_set_capacity >= remaining_volume:
            # Can assign the remaining volume in one shot
            # Max-min fairness in allocation
            # Increase the assignments to all links in the link set, equally
            # in every round and then stop when a link reaches capacity/
            rate_of_allocation = 3000
            remaining_volume_ = remaining_volume
            while remaining_volume_ > 0:
                for link in link_set:
                    if remaining_volume_ == 0 : break
                    assignment = None
                    if link not in link_assignments:
                        assignment = min([rate_of_allocation, remaining_volume_])
                        link_assignments[link] = assignment
                    elif ifspeeds[link] - link_assignments[link] > 0:
                        assignment = min([ifspeeds[link] - link_assignments[link],
                                          rate_of_allocation, remaining_volume_])
                        link_assignments[link] += assignment
                    else:
                        continue
                    remaining_volume_ -= assignment

            assert remaining_volume_ == 0
            remaining_volume = 0
        else:
            remaining_volume = remaining_volume - link_set_capacity
            for link in link_set:
                link_assignments[link] = ifspeeds[link]
                
        links_by_cost.pop(link_cost)

    for link in link_assignments:
        assert link_assignments[link] <= ifspeeds[link]
    assert sum(link_assignments.values()) == traffic_volume
    
    return link_assignments

def bin_pack_links_balanced(links, ifspeeds, traffic_volume, min_frac=0.1):
    ''' Given links with different costs and capacities,
    this method assigns billable bandwidth to the links such that
    the cost is minimzed and capacities are respected. This flavor of the
    bin packing is not optimal because it ensures that min_frac fraction of the total
    network capacity is spread evenly across all links. The goal of this is 
    to have the expensive links also be used since some clients
    are only connected via these links.
    '''
    assert min_frac <= 0.5
    link_assignments = {}
    pre_assigned  = 0
    for link in links:
        link_assignments[link] = min_frac * ifspeeds[link]
        pre_assigned += min_frac * ifspeeds[link]
    
    remaining_volume = traffic_volume - pre_assigned
    
    links_by_cost = {}
    for link in links:
        link_cost = cost_by_rtr(link)
        if link_cost not in links_by_cost:
            links_by_cost[link_cost] = []
        links_by_cost[link_cost].append(link)

    while remaining_volume > 0:
        # No links remain with available capacity
        if not links_by_cost:
            print "Cannot satisfy %f with this link set" % traffic_volume
            return None
        
        # Cheapest set of links
        link_set = sorted(links_by_cost.iteritems(), key=lambda x:x[0])[0][1]
        link_cost = sorted(links_by_cost.iteritems(), key=lambda x:x[0])[0][0]

        link_set_capacity = sum([ifspeeds[x]-link_assignments[x] for x in link_set])

        if link_set_capacity >= remaining_volume:
            # Can assign the remaining volume in one shot
            # Max-min fairness in allocation
            # Increase the assignments to all links in the link set, equally
            # in every round and then stop when a link reaches capacity/
            rate_of_allocation = 3000
            remaining_volume_ = remaining_volume
            while remaining_volume_ > 0:
                # print remaining_volume_, link_assignments
                for link in link_set:
                    if remaining_volume_ == 0 : break
                    assignment = 0
                    if ifspeeds[link] - link_assignments[link] > 0:
                        assignment = min([ifspeeds[link] - link_assignments[link],
                                          rate_of_allocation, remaining_volume_])
                        link_assignments[link] += assignment
                        try:
                            assert math.floor(link_assignments[link]) <= ifspeeds[link]
                        except:pdb.set_trace()
                    remaining_volume_ -= assignment

            assert remaining_volume_ == 0
            remaining_volume = 0
        else:
            remaining_volume = remaining_volume - link_set_capacity
            for link in link_set:
                link_assignments[link] = ifspeeds[link]
                
        links_by_cost.pop(link_cost)

    for link in link_assignments:
        try:
            assert math.floor(link_assignments[link]) <= ifspeeds[link]
        except:
            pdb.set_trace()

    try:
        assert math.ceil(sum(link_assignments.values())) >= math.floor(traffic_volume)
    except:
        pdb.set_trace()
    
    return link_assignments

def cascara_traffic_allocation(demand, init_fraction, links, ifspeeds, online=False,
                               alpha=None, beta=None):
    ''' init_fraction is the fraction of the total network capacity
    that we are willing to use for allocating traffic. It might be unfeasible
    to meet the traffic demand using this fraction, in which case we will update
    the fraction of the network capacity in use -- outside of the burst intervals.'''
    
    online_allocations = {}
    links_by_cost = {}
    linkq = PiorityQueue(ifspeeds)
    START_PRIO = len(demand)
    FREE_SLOTS = round(len(demand)*5/100.0)  - 1
    num_tses = len(demand)
    for link in links:
        linkq.add_link(link, FREE_SLOTS, START_PRIO)

    # Assigning the usable capacity fraction should be done effectively:
    # use links in an increasing order of their cost.
    total_capacity = float(sum([ifspeeds[x] for x in links]))
    capacity_fraction = init_fraction * total_capacity
    link_assignments = bin_pack_links(links, ifspeeds, capacity_fraction)
    if not link_assignments:
        print "Infeasible initial capacity fraction:", init_fraction
        return False, False
    
    sorted_tses = sorted(demand.keys())

    def allocate_timestep(ts, demand_in_ts, fraction):
        C_frac = fraction * total_capacity
        link_assignments = bin_pack_links(links, ifspeeds, C_frac)
        if demand_in_ts <= C_frac:
            for link in links:
                if link in link_assignments:
                    online_allocations[ts][link] = link_assignments[link]
                else:
                    online_allocations[ts][link] = 0
            return True
        else:
            over_demand = demand_in_ts - C_frac
            links_maxed_in_round = []
            while over_demand > 0:
                try:
                    linkname, slots, prio = linkq.pop_link()
                    links_maxed_in_round.append((linkname, slots, prio))
                except KeyError:
                    # print "%f Demand of timestamp %d not met yet and no links to max out" % \
                    #    (over_demand, ts)
                    for link, slots, prio in links_maxed_in_round:
                        linkq.add_link(link, slots, prio)
                    return False
                
                if linkname in link_assignments:
                    link_contrib = ifspeeds[linkname] - link_assignments[linkname]
                else:
                    link_contrib = ifspeeds[linkname]
                link_contrib = min([over_demand, link_contrib])
                over_demand = over_demand - link_contrib

            for link in links:
                if link in link_assignments:
                    online_allocations[ts][link] = link_assignments[link]
                else:
                    online_allocations[ts][link] = 0

            for link, slots, prio in links_maxed_in_round:
                linkq.add_link(link, slots - 1, prio - 1)
                online_allocations[ts][link] = ifspeeds[link]

            assert sum(online_allocations[ts].values()) >= demand_in_ts

            for link in online_allocations[ts]:
                assert online_allocations[ts][link] <= ifspeeds[link]
            return True
        
        assert False

    def inc_frac(frac, value):
        frac = frac + value
        if frac > 1:
            frac = 1
        return frac
    
    fraction = init_fraction
    count = 0
    week_count = 1
    init_bca = linkq.get_bca(link_assignments)
    for ts in sorted_tses:
        demand_in_ts = demand[ts]
        C_frac = fraction * total_capacity
        new_link_assignments = bin_pack_links(links, ifspeeds, C_frac)
        count += 1
        if count % 2016 == 0 and online:
            # one week is complete
            if linkq.get_bca(new_link_assignments) < init_bca * (4-week_count)/4.0:
                fraction = inc_frac(fraction, alpha)
            week_count += 1
                
        online_allocations[ts] = {}
        while not allocate_timestep(ts, demand_in_ts, fraction):
            # print "Allocation attempt in ts %d failed with capacity fraction %f" % (ts, fraction)
            # print "Demand was %f and burstable capacity was %f" % \
            #    (demand_in_ts, linkq.get_bca(new_link_assignments))
            if not online:
                return None, None
            fraction = inc_frac(fraction, beta)
            print week_count, fraction    

    online_allocations_by_rtr = {}
    for ts in online_allocations:
            for rtr in online_allocations[ts]:
                if rtr not in online_allocations_by_rtr:
                    online_allocations_by_rtr[rtr] = []
                online_allocations_by_rtr[rtr].append(online_allocations[ts][rtr])

    return online_allocations_by_rtr, fraction

def exponential_weighted_moving_average_demand_prediction(demands, beta=1):
    '''
    Exponential weighted moving average prediction of traffic
    on peering links or overall demand as suggested by Goldenberg et. al 
    in "Optimizing Cost and Performance for Multihoming" in SIGCOMM 2004.
    '''
    sorted_tses = sorted(demands.keys())
    prediction = 0
    predictions = {}
    for ts in sorted_tses:
        index = sorted_tses.index(ts)
        if index + 1 >= len(sorted_tses):
            continue
        next_ts = sorted_tses[index + 1]
        current_demand = demands[ts]
        next_prediction = beta * current_demand  + (1-beta) * prediction
        predictions[next_ts] = next_prediction
        prediction = next_prediction
        
    return predictions

def get_demand_func_time(post_op, pre_op):
    demand = {}
    for ts in pre_op:
        if ts not in demand:
            demand[ts] = 0
        demand[ts] = sum(pre_op[ts].values())
        try:
            assert round(sum(post_op[ts].values())) >= round(demand[ts])
        except:
            pdb.set_trace()
    return demand


def read_contiguous_traffic_demand(cluster_no):
    all_demand = {}
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
    ]:
        pre_op, post_op, pre_op_by_rtr_list, post_op_by_rtr_list,  link_set =\
                    read_allocation(month, cluster_no)
        demand = get_demand_func_time(post_op, pre_op)
        all_demand.update(demand)
    return all_demand
        
def offline_V_0(all_demand, cluster, end_ts, I, links, ifspeeds):
    sorted_tses = sorted(all_demand.keys())
    end_ts_index = sorted_tses.index(end_ts)
    start_ts_index = end_ts_index - I
    assert end_ts_index > 0
    start_ts_index = max([0, start_ts_index])
    assert start_ts_index >= 0

    start_ts = sorted_tses[start_ts_index]
    
    relevant_demand = {}
    for ts in sorted_tses:
        if ts >= start_ts and ts <= end_ts:
            relevant_demand[ts] = all_demand[ts]

    _, _, V_0 = sigcomm04_heuristic_offline(relevant_demand, links, ifspeeds)
    return V_0
    
def gia_online(demand, all_demand, links, ifspeeds, V_0, cluster):
    ''' This is the GIA-online algorithm from "Optimizing
    Cost and Performance for Multihoming" paper from SIGCOMM 2004 by
    Goldenberg et.al.'''
    
    predicted_total_demand = exponential_weighted_moving_average_demand_prediction(demand)
    def select_random_links_with_slots(links, slots, p_k, k):
        link_subset = [link for link in links if slots[link] > 0]
        if len(link_subset) == 0:
            return []
        elif len(link_subset) >= k:
            return random.sample(link_subset, k)
        else:
            return link_subset
    
    sorted_tses = sorted(demand.keys())
    I = len(sorted_tses)
    f = min([len(links) * 0.05, 1])
    V_0 = np.percentile(demand.values(), (1-f)*100)
    delta = 0.01
    slots = {}
    FREE_SLOTS = round(len(demand)*5/100.0)  - 1
    
    for link in links:
        slots[link] = FREE_SLOTS

    def assign_predicted_flow(flow, V_0, p_k, links, slots):
        if flow <= V_0: return slots, True
        demand_met = False
        for j in range(1, len(links) + 1):
            peak_links = select_random_links_with_slots(links, slots, p_k, j)
            burst_cap = sum([ifspeeds[x] for x in peak_links])
            burst_cap_deduct = sum([p_k[x] for x in peak_links])
            burst_cap = V_0 + burst_cap - burst_cap_deduct
            if burst_cap >= flow:
                demand_met = True
                break
                
        if demand_met:
            for link in peak_links:
                slots[link] -= 1
        else:
            return slots, False
        return slots, True

    previous_demand = None
    previous_p_k = {}
    for link in links:
        previous_p_k[link] = 0

    count = 0 
    for ts in sorted_tses:
        count += 1
        if count % 2016 == 0:
            V_0_prime = offline_V_0(all_demand, cluster, ts, I, links, ifspeeds)
            if V_0_prime > V_0:
                #print "increasing v_0"
                V_0 = 1.05* V_0
            
        margin = 0.05 * V_0
        demand_in_ts = demand[ts]
        if ts in predicted_total_demand:
            predicted_demand_in_ts = predicted_total_demand[ts]
        else:
            predicted_demand_in_ts = demand_in_ts
        link_costs = [(link, cost_by_rtr(link)) for link in links]
        link_costs = sorted(link_costs, key = lambda x: x[1])
        remaining = V_0 + margin * len(links)
        p_k = previous_p_k.copy()
        for link, cost in link_costs:
            if remaining <= 0:
                continue
            cap = ifspeeds[link]
            if remaining < cap:
                p_k[link] = max([remaining, previous_p_k[link]])
                remaining = 0
            else:
                p_k[link] = cap
                remaining = remaining - cap

        updated_p_k = {}
        for link in p_k:
            update =  max([0, p_k[link] - margin])
            updated_p_k[link] = max([previous_p_k[link], update])
            assert updated_p_k[link] >= previous_p_k[link]

        assert sum(updated_p_k.values()) >= V_0
        # assign the predicted flow
        slots, feasible = assign_predicted_flow(demand_in_ts, sum(updated_p_k.values()),
                                                updated_p_k, links, slots)
        if not feasible:
            return None, None
            
        previous_p_k = updated_p_k

    cost = sum([updated_p_k[link]*cost_by_rtr(link) for link in updated_p_k])
    return updated_p_k, cost

def entact(demand, all_demand, links, ifspeeds):
    '''
    Entact algorithm from https://www.usenix.org/legacy/event/nsdi10/tech/full_papers/zhang.pdf
    The core objective function minimizes the cost in every 5-minute time window.
    Cost is calculated as a product of link price and traffic assigned to it.
    This is essentially the greedy objective which can be minimized by assigning
    as much flow as is possible to the cheapest link, then the next cheapest and so on.
    '''
    sorted_tses = sorted(demand.keys())
    online_allocations = {}
    for ts in sorted_tses:
        online_allocations[ts] = {}
        demand_in_ts = demand[ts]
        link_assignments = bin_pack_links(links, ifspeeds, demand_in_ts)
        if not link_assignments:
            pdb.set_trace()
        for link in links:
            if link in link_assignments:
                online_allocations[ts][link] = link_assignments[link]
            else:
                online_allocations[ts][link] = 0

    online_allocations_by_rtr = {}
    for ts in online_allocations:
            for rtr in online_allocations[ts]:
                if rtr not in online_allocations_by_rtr:
                    online_allocations_by_rtr[rtr] = []
                online_allocations_by_rtr[rtr].append(online_allocations[ts][rtr])

    return online_allocations_by_rtr

def cascara_traffic_allocation_stable(demand, init_fraction, links, ifspeeds, online=False,
                                      alpha=None, beta=None, max_rate=50):
    ''' init_fraction is the fraction of the total network capacity
    that we are willing to use for allocating traffic. It might be unfeasible
    to meet the traffic demand using this fraction, in which case we will update
    the fraction of the network capacity in use -- outside of the burst intervals.'''
    allocation_step = max_rate * 5 * 60 # 10G for 5 minutes
    online_allocations = {}
    links_by_cost = {}
    linkq = PiorityQueue(ifspeeds)
    START_PRIO = len(demand)
    FREE_SLOTS = round(len(demand)*5/100.0)  - 1
    num_tses = len(demand)
    for link in links:
        linkq.add_link(link, FREE_SLOTS, START_PRIO)

    # Assigning the usable capacity fraction should be done effectively:
    # use links in an increasing order of their cost.
    total_capacity = float(sum([ifspeeds[x] for x in links]))
    capacity_fraction = init_fraction * total_capacity
    link_assignments = bin_pack_links(links, ifspeeds, capacity_fraction)
    if not link_assignments:
        # print "Infeasible initial capacity fraction:", init_fraction
        return False, False
    
    sorted_tses = sorted(demand.keys())

    def allocate_timestep(ts, demand_in_ts, fraction, previous_ts):
        C_frac = fraction * total_capacity
        # print ts, C_frac
        link_assignments = bin_pack_links(links, ifspeeds, C_frac)
        if demand_in_ts <= C_frac:
            for link in links:
                if link in link_assignments:
                    online_allocations[ts][link] = link_assignments[link]
                else:
                    online_allocations[ts][link] = 0
            return True
        else:
            over_demand = demand_in_ts - C_frac
            allocations_previous_timestep = {}
            if previous_ts in online_allocations:
                allocations_previous_timestep = online_allocations[previous_ts]
            links_maxed_in_round = []
            link_to_contrib = {}
            while over_demand > 0:
                try:
                    linkname, slots, prio = linkq.pop_link()
                    links_maxed_in_round.append((linkname, slots, prio))
                except KeyError:
                    # print "%f Demand of timestamp %d not met yet and no links to max out" % \
                    #    (over_demand, ts)
                    for link, slots, prio in links_maxed_in_round:
                        linkq.add_link(link, slots, prio)
                    return False
                if linkname in allocations_previous_timestep:
                    previous_alloc = allocations_previous_timestep[linkname]
                elif linkname in link_assignments:
                    previous_alloc = link_assignments[linkname]
                else:
                    previous_alloc = 0

                # link_contrib <= ifspeed
                # link_contrib <= previous_alloc + alloc_timestep
                link_contrib = min([previous_alloc + allocation_step, ifspeeds[linkname]])
                # If this link has some assignment by bin packing,
                # it could have been increased by the allocation step
                # so map that to the link's contribution.
                # however change the contribution to be subtracted
                # from the over_demand to the excess over earlier
                # allocation -- over_demand already counts the
                # earlier allocation.
                if linkname in link_assignments:
                    if link_contrib > link_assignments[linkname]:
                        link_contrib = link_contrib - link_assignments[linkname]
                    else:
                        link_contrib = 0

                # link_contrib <= over_demand                    
                link_contrib = min([over_demand, link_contrib])                    
                assert link_contrib >= 0
                over_demand = over_demand - link_contrib
                link_to_contrib[linkname] = link_contrib
                
            for link in links:
                if link in link_assignments:
                    online_allocations[ts][link] = link_assignments[link]
                else:
                    online_allocations[ts][link] = 0

            for link, slots, prio in links_maxed_in_round:
                linkq.add_link(link, slots - 1, prio - 1)
                if link not in online_allocations[ts]:
                    online_allocations[ts][link] = 0
                online_allocations[ts][link] += link_to_contrib[link]

            assert sum(online_allocations[ts].values()) >= demand_in_ts

            for link in online_allocations[ts]:
                assert online_allocations[ts][link] <= ifspeeds[link]
            return True
        
        assert False

    def inc_frac(frac, value):
        frac = frac + value
        if frac > 1:
            frac = 1
        return frac
    
    fraction = init_fraction
    count = 0
    week_count = 1
    init_bca = linkq.get_bca(link_assignments)
    previous_ts = None
    for ts in sorted_tses:
        demand_in_ts = demand[ts]
        C_frac = fraction * total_capacity
        new_link_assignments = bin_pack_links(links, ifspeeds, C_frac)
        count += 1
        if count % 2016 == 0 and online:
            # one week is complete
            if linkq.get_bca(new_link_assignments) < init_bca * (4-week_count)/4.0:
                fraction = inc_frac(fraction, alpha)
            week_count += 1
                
        online_allocations[ts] = {}
        while not allocate_timestep(ts, demand_in_ts, fraction, previous_ts):
            if not online:
                return None, None
            fraction = inc_frac(fraction,  beta)
        previous_ts = ts
        
    online_allocations_by_rtr = {}
    for ts in online_allocations:
            for rtr in online_allocations[ts]:
                if rtr not in online_allocations_by_rtr:
                    online_allocations_by_rtr[rtr] = []
                online_allocations_by_rtr[rtr].append(online_allocations[ts][rtr])

    return online_allocations_by_rtr, fraction

def get_links(peer, all_links, pop=None):
    ''' get links at a pop and belonging to 
    a BGP peer.'''
    relevant_links = []
    for link in all_links:
        if peer in link:
            if pop:
                if "-%s-" % pop in link:
                    relevant_links.append(link)
            else:
                relevant_links.append(link)
                
    return relevant_links

def get_client_to_link_maps(ts, client_maps, client_to_primary_peer,
                            client_to_primary_pop, client_to_pop_latency,
                            all_links):
    client_to_pop_latency_tses = sorted(client_to_pop_latency.keys())
    
    def get_lowest_latency_links(client, ts, primary_pop=None,enable_transit=False):
        ts_prev = None
        for ts_iter in client_to_pop_latency_tses:
            if ts_iter >= int(ts):
                ts_end = ts_iter
                break
            ts_prev = ts_iter

        best_pop = None
        all_measured_pops = {}
        lookback = 0
        for ts_iter in reversed(client_to_pop_latency_tses):
            # Python weirdness: if ts_prev is None, the following condition retursn true
            if ts_iter > ts_prev: continue
            lookback += 1
            if lookback > 10: break
            if client in client_to_pop_latency[ts_iter]:
                measured_pops = client_to_pop_latency[ts_iter][client]
                for pop, lat in measured_pops:
                    if pop not in all_measured_pops:
                        # keeping the most recent measurement
                        all_measured_pops[pop] = lat
        better_pops = []
        for pop1 in all_measured_pops:
            for pop2 in all_measured_pops:
                if pop1 == pop2: continue
                if pop1 in primary_pop or pop2 in primary_pop:
                    if abs(all_measured_pops[pop1] - all_measured_pops[pop2]) <= 10:
                        better_pops.append(pop1)
                        better_pops.append(pop2)

        if better_pops:
            better_pops = list(set(better_pops))                        
            better_pops = list(set(better_pops).union(primary_pop))
        else:
            better_pops = primary_pop
        client_primary_peers = client_to_primary_peer[client]
        relevant_links = []
        for bpop in better_pops:
            for peer in client_primary_peers:
                relevant_links.extend(get_links(peer, all_links, pop=bpop))

        return relevant_links
    
    with open(CLIENT_TO_LINK_MAPS) as fi:
        client_to_link_maps_ = json.load(fi)

    if client_maps == "current":
        return client_to_link_maps_
    
    elif client_maps == "all":
        client_to_link_maps = {}
        for client in client_to_link_maps_:
            client_to_link_maps[client] = []
            for peer in client_to_primary_peer[client]:
                client_to_link_maps[client].extend(peer_to_links[peer])
        return client_to_link_maps
    
    elif client_maps == "all-transit":
        # fun things here
        client_to_link_maps = {}
        for client in client_to_link_maps_:
            relevant_links = []
            for peer in client_to_primary_peer[client]:
                relevant_links.extend(peer_to_links[peer])
            # Address space overlap is such that these links are also
            # feasible choices
            client_to_link_maps[client] = relevant_links
            
        return client_to_link_maps

    elif client_maps == "pop":
        client_to_link_maps = {}
        for client in client_to_link_maps_:
            relevant_links = []
            for peer in client_to_primary_peer[client]:
                for ppop in client_to_primary_pop[client]:
                    relevant_links.extend(get_links(peer, all_links, pop=ppop))
            client_to_link_maps[client] = relevant_links
        return client_to_link_maps
    
    elif client_maps == "pop-transit":
        client_to_link_maps = {}
        for client in client_to_link_maps_:
            relevant_links = []
            for peer in client_to_primary_peer[client]:
                for ppop in client_to_primary_pop[client]:
                    relevant_links.extend(get_links(peer, all_links, pop=ppop))
            client_to_link_maps[client] = relevant_links

        return client_to_link_maps
    
    elif client_maps == "pop2":
        client_to_link_maps = {}
        for client in client_to_link_maps_:
            client_to_link_maps[client] =\
                get_lowest_latency_links(client, ts,
                                        primary_pop=client_to_primary_pop[client])
        return client_to_link_maps

    elif client_maps == "pop2-transit":
        client_to_link_maps = {}
        for client in client_to_link_maps_:
            client_to_link_maps[client] = get_lowest_latency_links(client, ts,
                                                                   primary_pop=client_to_primary_pop[client],
                                                                   enable_transit=True)
        return client_to_link_maps
    
    
def cascara_traffic_allocation_latency_sensitive(demand, init_fraction, links,
                                                 client_to_primary_peer,
                                                 client_to_primary_pop,
                                                 client_to_pop_latency,
                                                 ifspeeds, online=False,
                                                 alpha=None, beta=None,
                                                 client_maps="current"):
    print "Client map strategy:", client_maps
    max_rate = 40
    limit_per_link = 0.2
    print "Limit per link", limit_per_link
    allocation_step = max_rate * 5 * 60 # 10G for 5 minutes    
    online_allocations = {}
    links_by_cost = {}
    linkq = PiorityQueue(ifspeeds)
    START_PRIO = len(demand)
    FREE_SLOTS = round(len(demand)*5/100.0)  - 1
    num_tses = len(demand)

                
    # Assigning the usable capacity fraction should be done effectively:
    # use links in an increasing order of their cost.
    total_capacity = float(sum([ifspeeds[x] for x in links]))
    capacity_fraction = init_fraction * total_capacity
    
    sorted_tses = sorted(demand.keys())
    
    env = Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    
    def allocate_timestep_latency_sensitive(ts, previous_ts, demand_in_ts,
                                            fraction, min_frac=0.2):
        
        client_to_link_maps = get_client_to_link_maps(ts, client_maps,
                                                      client_to_primary_peer,
                                                      client_to_primary_pop,
                                                      client_to_pop_latency,
                                                      links)
        # demand_in_ts: {client: demand}
        # client_to_link_maps: {client: peers}
        # link_allocations: {link: allocation}
        C_frac = fraction * total_capacity
        over_demand = 0
        burst_link_allocs = {}
        links_maxed_in_round = []
        if sum(demand_in_ts.values()) >= C_frac:
            # the allocation will fail because C_frac isn't enough
            # We have to augment some links
            print "Demand is higher than C_frac", C_frac, sum(demand_in_ts.values())
            over_demand = sum(demand_in_ts.values()) - C_frac
            while over_demand > 0:
                try:
                    linkname, slots, prio = linkq.pop_link()
                    links_maxed_in_round.append((linkname, slots, prio))
                except KeyError:
                    # Ran out of links to max out, add the ones popped
                    # off the priority queue and return
                    for link, slots, prio in links_maxed_in_round:
                        linkq.add_link(link, slots, prio)
                        print "burst_allocations_timestep: Ran out of links to max out", ts
                        return False
                link_contrib  = min(ifspeeds[linkname], over_demand)
                burst_link_allocs[linkname] = link_contrib
                over_demand = over_demand - link_contrib

        assert over_demand == 0
        C_frac  = sum(demand_in_ts.values())
        
        model = Model("lp", env=env)
        variables = {}
        objective = 0
        total_allocation = 0
        per_link_vars = {}
        # i is the client + metro
        for i in demand_in_ts:
            variables[i] = {}
            if len(client_to_link_maps[i]) == 0:
                # This is a stop gap because there are no feasible links
                # to this client and that needs to be fixed
                # print "No links for clients", i, "mapping to all links by the same peer"
                client_to_link_maps[i] = get_links(client_to_primary_peer[i][0], links, pop=None)
            try:
                assert len(client_to_link_maps[i]) >= 1, "too few links"
            except:
                pdb.set_trace()
            # j is the link (peer + rtr)
            for j in links:
                variables[i][j] = model.addVar(lb=0, name="x_%s_%s" % (i,j))
                if j not in client_to_link_maps[i]:
                    # this link is there in the global graph but the link
                    # selection done by get_client_to_link_maps
                    # has not considered it "relevant" for the client_map
                    # strategy. To ensure the link doesn't have
                    # a non-zero allocation on it, adding a special constraint.
                    model.addConstr(variables[i][j] <= 0, name="%s-%s-positive" % (i, j))
                    
                objective += cost_by_rtr(j) * variables[i][j]
                total_allocation += variables[i][j]
                if j not in per_link_vars:
                    per_link_vars[j] = 0
                per_link_vars[j] += variables[i][j]
            # allocations from client i to any links should
            # be at least as high as the demand to the client
            if i in demand_in_ts:
                model.addConstr(sum(variables[i].values()) >= demand_in_ts[i],
                                name="demand_to_%s" % i)
            else:
                model.addConstr(sum(variables[i].values()) >= 0)
                      
        model.update()

        # trafic towards all clients from a link
        # should not send more than the allocation amount
        # predetermined for this link
        for link in per_link_vars:
            if link in  burst_link_allocs:
                model.addConstr(per_link_vars[link] <= burst_link_allocs[link],
                                name="burst_link_%s" % link)
            else:
                model.addConstr(per_link_vars[link] <= limit_per_link * ifspeeds[link],
                                name="link_capzacity_%s" % link)

        model.setObjective(objective)
        model.addConstr(total_allocation <= C_frac, name="C_frac")
        
        try:
            model.optimize()
        except error.SolverError as e:
            print e
            print "Gurobi failed for timestamp", ts, fraction
            return 0

        if model.Status == GRB.INFEASIBLE:
            print "Gurobi model is infeasible", ts, fraction
            
            for link, slots, prio in links_maxed_in_round:
                linkq.add_link(link, slots, prio)
            model_c = model.copy()
            model_c.computeIIS()
            for c in model_c.getConstrs():
                if c.IISConstr: print('%s' % c.constrName)
            pdb.set_trace()
            return 0

        for link, slots, prio in links_maxed_in_round:
            linkq.add_link(link, slots, prio)        
        
        for i in variables:
            for j in variables[i]:
                if j not in online_allocations[ts]:
                    online_allocations[ts][j] = 0
                online_allocations[ts][j] += variables[i][j].x
                                                    
        return 1
                                
    def inc_frac(frac, value):
        print "Inc frac:", frac, frac+value
        frac = frac + value
        if frac >= 1:
            frac = 1
        return frac
    
    fraction = init_fraction
    count = 0
    week_count = 1

    previous_ts = None

    link_to_metros = {}
    print "Fraction initial", fraction
    for ts in sorted_tses:
        print "TIMESTAMP FOR LP:", ts
        demand_in_ts = demand[ts]
        C_frac = fraction * total_capacity
        count += 1
        online_allocations[ts] = {}
        alloc_found = False
        while not alloc_found:
            return_status = allocate_timestep_latency_sensitive(ts, previous_ts, demand_in_ts,
                                                                fraction)
            if return_status!= 1 and not online:
                print "Allocation failed at ts", ts, C_frac
                return None, None
            elif return_status != 1:
                fraction = inc_frac(fraction, beta)
                print week_count, fraction
                continue
            # allocation successful
            alloc_found = True
            
        previous_ts = ts
        
    online_allocations_by_rtr = {}
    for ts in online_allocations:
            for rtr in online_allocations[ts]:
                if rtr not in online_allocations_by_rtr:
                    online_allocations_by_rtr[rtr] = []
                online_allocations_by_rtr[rtr].append(online_allocations[ts][rtr])

    return online_allocations_by_rtr, fraction
