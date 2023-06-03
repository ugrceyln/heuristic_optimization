import math
import random
import time
import numpy
import pickle
import collections

from itertools import permutations

class Circle:

    def __init__(self, x=0.0, y=0.0, r=1.0):

        self.x = x
        self.y = y
        self.r = r

def get_search_space_domain(input_circle_list):

    all_domain = permutations(input_circle_list, len(input_circle_list))
    all_unique_domain = list(set(all_domain))
    seen = set()
    all_unique_domain_removed_reversed = [x for x in all_unique_domain if tuple(x[::-1]) not in seen and not seen.add(tuple(x))]

    search_space = all_unique_domain_removed_reversed

    return search_space

def arrange_circles_and_get_min_width_box(input_circles):

    l_circle_obj = []

    for i, circle_r in enumerate(input_circles):
        l_circle_obj.append(Circle(r=circle_r))

    base_circle = l_circle_obj[0]

    mr_points = []
    ml_points = []
    xs = []
    ys = []
    rs = []

    mr_points.append(2 * base_circle.r)
    ml_points.append(0)
    xs.append(base_circle.r)
    ys.append(base_circle.r)
    rs.append(base_circle.r)
    min_width = 2 * base_circle.r     # max(input_circles) * len(input_circles) * 2

    for circle in l_circle_obj[1:]:
        y2 = circle.r
        r2 = circle.r

        l_x2 = []
        for x1, y1, r1 in zip(xs, ys, rs):
            x2 = math.sqrt((r1 + r2) ** 2 - (y2 - y1) ** 2) + x1
            l_x2.append(x2)
            mr_points.append(x2 + r2)
            ml_points.append(x2 - r2)

        xs.append(max(l_x2))
        ys.append(y2)
        rs.append(r2)

        temp_min_width = abs(min(ml_points)) + max(mr_points)

        if temp_min_width > min_width:
            min_width = temp_min_width

    return min_width

def get_brute_force_solution(search_space):

    l_width_box = []

    for circles in search_space:
        width_box = arrange_circles_and_get_min_width_box(circles)
        l_width_box.append(width_box)

    min_width_box = min(l_width_box)
    min_width_index = l_width_box.index(min_width_box)
    arrangement_circles = search_space[min_width_index]

    return min_width_box, arrangement_circles

def get_greedy_solution(input_circle_list):

    sorted_input_circles = sorted(input_circle_list)
    min_width_box = arrange_circles_and_get_min_width_box(sorted_input_circles)

    min_width_box=min_width_box; arrangement_circles=sorted_input_circles

    return min_width_box, arrangement_circles

def generate_neighborhood_structure_for_ils_1_opt(current_solution):

    def swap_position(list_, pos1, pos2):

        list_[pos1], list_[pos2] = list_[pos2], list_[pos1]
        return list_

    l_curr_solution = list(current_solution)
    l_curr_solution_ori = l_curr_solution.copy()
    l_neighbors = []
    len_curr_sol = len(l_curr_solution_ori)

    for i in range(0, len_curr_sol-1):
        for j in range(i+1, len_curr_sol):
            current_solution = l_curr_solution_ori.copy()
            neighbor = swap_position(current_solution, i, j)
            l_neighbors.append(neighbor)

    if l_neighbors.__contains__(l_curr_solution_ori):
        l_neighbors.remove(l_curr_solution_ori)

    l_neighbors = [list(x) for x in set(tuple(x) for x in l_neighbors)]

    return l_neighbors

def get_iterated_local_search_solution(search_space, iteration_num=100):

    best_solution_width = 2 * max(list(search_space[0])) * len(list(search_space[0]))
    best_solution = list(search_space[0])

    for iter_num in range(iteration_num):

        init_solution = list(random.choice(search_space))
        init_width_box = arrange_circles_and_get_min_width_box(init_solution)
        current_solution = init_solution

        local_best_solution = init_solution
        local_best_solution_width = init_width_box

        while True:

            neighbors = generate_neighborhood_structure_for_ils_1_opt(current_solution)

            l_neighbor_width = []
            for neighbor in neighbors:
                min_width_box = arrange_circles_and_get_min_width_box(neighbor)
                l_neighbor_width.append(min_width_box)

            local_best_neighbor_width = min(l_neighbor_width)
            local_best_neighbor = neighbors[l_neighbor_width.index(local_best_neighbor_width)]

            if local_best_neighbor_width < local_best_solution_width:
                local_best_solution_width = local_best_neighbor_width
                current_solution = local_best_neighbor.copy()
                local_best_solution = local_best_neighbor

            else:
                break

        if local_best_solution_width < best_solution_width:
            best_solution = local_best_solution
            best_solution_width = local_best_solution_width

    min_width_box = best_solution_width
    arrangement_circles = best_solution

    return min_width_box, arrangement_circles

def generate_neighborhood_for_vns_with_2_opt(current_solution, k):

    def swap_position(list_, pos1, pos2):

        list_[pos1], list_[pos2] = list_[pos2], list_[pos1]
        return list_

    for i in range(k):
        four_indices = numpy.random.choice(range(len(current_solution)), size=4, replace=False)
        current_solution = swap_position(current_solution, four_indices[0], four_indices[1])
        current_solution = swap_position(current_solution, four_indices[2], four_indices[3])

    return current_solution

def local_search_for_vns_with_2_opt(current_solution):

    def swap_position(list_, pos1, pos2):

        list_[pos1], list_[pos2] = list_[pos2], list_[pos1]
        return list_

    l_curr_solution = list(current_solution)
    l_curr_solution_ori = l_curr_solution.copy()
    l_neighbors = []
    len_curr_sol = len(l_curr_solution_ori)

    for i in range(len_curr_sol-1):

        for j in range(i+1, len_curr_sol):

            current_solution = l_curr_solution_ori.copy()
            neighbor_1_opt = swap_position(current_solution, i, j)

            indices = list(range(0, len(current_solution)))
            indices.remove(i)
            indices.remove(j)
            neighbor_1_opt_ori = neighbor_1_opt.copy()

            for k in indices[0:-1]:
                index_k = indices.index(k)
                for l in indices[index_k+1:]:
                    neighbor_1_opt = neighbor_1_opt_ori.copy()
                    neighbor = swap_position(neighbor_1_opt, k, l)
                    l_neighbors.append(neighbor)

    if l_neighbors.__contains__(l_curr_solution_ori):
        l_neighbors.remove(l_curr_solution_ori)

    l_neighbors = [list(x) for x in set(tuple(x) for x in l_neighbors)]

    return l_neighbors

def get_variable_neighborhood_search_solution(search_space, iteration_num=50, neighborhoods=5):

    best_solution_width = 2 * max(list(search_space[0])) * len(list(search_space[0]))
    best_solution = list(search_space[0])

    for iter_num in range(iteration_num):

        init_solution = list(random.choice(search_space))
        init_width_box = arrange_circles_and_get_min_width_box(init_solution)
        current_solution = init_solution

        local_best_solution = init_solution
        local_best_solution_width = init_width_box

        k = 1
        while k < neighborhoods:

            neighbor = generate_neighborhood_for_vns_with_2_opt(current_solution, k)
            neighbors = local_search_for_vns_with_2_opt(neighbor)

            l_neighbor_width = []
            for neighbor in neighbors:
                min_width_box = arrange_circles_and_get_min_width_box(neighbor)
                l_neighbor_width.append(min_width_box)

            local_best_neighbor_width = min(l_neighbor_width)
            local_best_neighbor = neighbors[l_neighbor_width.index(local_best_neighbor_width)]

            if local_best_neighbor_width < local_best_solution_width:
                local_best_solution_width = local_best_neighbor_width
                current_solution = local_best_neighbor.copy()
                local_best_solution = local_best_neighbor

            else:
                k+=1

        if local_best_solution_width < best_solution_width:
            best_solution = local_best_solution
            best_solution_width = local_best_solution_width

    min_width_box=best_solution_width
    arrangement_circles=best_solution

    return min_width_box, arrangement_circles

def get_neighborhood_for_sa_with_2_opt(current_solution):

    def swap_position(list_, pos1, pos2):

        list_[pos1], list_[pos2] = list_[pos2], list_[pos1]
        return list_

    four_indices = numpy.random.choice(range(len(current_solution)), size=4, replace=False)
    current_solution = swap_position(current_solution, four_indices[0], four_indices[1])
    current_solution = swap_position(current_solution, four_indices[2], four_indices[3])

    return current_solution

def get_simulated_annealing_solution(search_space, T=1000000, r=0.9, lb_T = 0.1, MAX_TRIES=100):

    init_solution = list(search_space[0])
    best_solution = init_solution.copy()
    best_solution_width = arrange_circles_and_get_min_width_box(best_solution)

    tries = 0
    while tries < MAX_TRIES:

        init_solution = list(random.choice(search_space))
        current_solution = init_solution
        current_solution_width = arrange_circles_and_get_min_width_box(current_solution)

        t = T

        while t > lb_T:

            neighbor_solution = get_neighborhood_for_sa_with_2_opt(current_solution)
            neighbor_width_box = arrange_circles_and_get_min_width_box(list(neighbor_solution))

            if neighbor_width_box < current_solution_width:
                current_solution_width = neighbor_width_box

            else:
                q = random.uniform(0, 1)
                rand_change_val = (math.e ** ((current_solution_width - neighbor_width_box)/T))

                if q < rand_change_val:
                    current_solution = neighbor_solution

            t = t * r

        if current_solution_width < best_solution_width:
            best_solution_width = current_solution_width
            best_solution = current_solution

        tries += 1

    min_width_box=best_solution_width
    arrangement_circles=best_solution

    return min_width_box, arrangement_circles

def swap_position(list_, pos1, pos2):

    list_[pos1], list_[pos2] = list_[pos2], list_[pos1]
    return list_

def generate_neighbor_swap_lists_as_1_opt_for_ts(curr_solution):

    len_curr_sol = len(curr_solution)
    tabu_structure = {}
    for i in range(0, len_curr_sol-1):
        for j in range(i+1, len_curr_sol):
            tabu_structure[(i, j)] = 0

    return tabu_structure

def get_swaped_indices(l_curr_solution_ori, best_neighbor):

    len_curr_sol = len(l_curr_solution_ori)

    for i in range(0, len_curr_sol - 1):
        for j in range(i + 1, len_curr_sol):
            current_solution = l_curr_solution_ori.copy()
            neighbor = swap_position(current_solution, i, j)

            if neighbor == best_neighbor:
                return (i, j)

def control_is_tabu(tabu_structure, best_neighbor_key):

    value_best = tabu_structure.get(best_neighbor_key)

    if value_best != 0:
        return True

    return False

def get_best_neighbor_for_ts_1_opt(tabu_structure, current_solution, best_solution_width, tabu_tenure=5):

    l_curr_solution_ori = list(current_solution).copy()
    len_curr_sol = len(l_curr_solution_ori)

    d_neighbors = {}
    for i in range(0, len_curr_sol-1):
        for j in range(i+1, len_curr_sol):
            current_solution = l_curr_solution_ori.copy()
            neighbor = swap_position(current_solution, i, j)
            min_width_box = arrange_circles_and_get_min_width_box(neighbor)
            d_neighbors[(i, j)] = min_width_box

    sorted_d_neighbors = sorted(d_neighbors.items(), key=lambda kv: kv[1])
    best_neighbor_key = sorted_d_neighbors[0][0]
    value_best = d_neighbors.get(best_neighbor_key)
    current_solution = l_curr_solution_ori.copy()
    current_solution = swap_position(current_solution, list(best_neighbor_key)[0], list(best_neighbor_key)[1])

    if value_best < best_solution_width:
        best_solution_width = value_best

        for key in tabu_structure.keys():
            if key == best_neighbor_key:
                tabu_structure[best_neighbor_key] = tabu_tenure
            else:
                if tabu_structure.get(key) > 0:
                    tabu_structure[key] -= 1
    else:
        count = 1
        while (control_is_tabu(tabu_structure, best_neighbor_key)):
            best_neighbor_key = sorted_d_neighbors[count][0]
            count += 1

        current_solution = l_curr_solution_ori.copy()
        current_solution = swap_position(current_solution, list(best_neighbor_key)[0], list(best_neighbor_key)[1])
        best_solution_width = arrange_circles_and_get_min_width_box(current_solution)

        for key in tabu_structure.keys():
            if key == best_neighbor_key:
                tabu_structure[best_neighbor_key] = tabu_tenure
            else:
                if tabu_structure.get(key) > 0:
                    tabu_structure[key] -= 1

    return best_solution_width, current_solution

def get_tabu_search_solution(search_space, MAX_TRIES=100, ITER=20):

    best_solution_width = 2 * max(list(search_space[0])) * len(list(search_space[0]))
    best_solution = list(search_space[0])

    for tries in range(MAX_TRIES):

        init_solution = list(random.choice(search_space))
        current_solution = init_solution

        tabu_structure = generate_neighbor_swap_lists_as_1_opt_for_ts(init_solution)
        iter = 0

        while iter < ITER:

            local_best_neighbor_width, local_best_neighbor = get_best_neighbor_for_ts_1_opt(tabu_structure, current_solution, best_solution_width)

            if local_best_neighbor_width < best_solution_width:
                best_solution_width = local_best_neighbor_width
                current_solution = local_best_neighbor.copy()
                best_solution = local_best_neighbor

            else:
                iter += 1

    min_width_box = best_solution_width
    arrangement_circles = best_solution

    return min_width_box, arrangement_circles

def initialize_population_and_other_params(search_space):

    search_space_len = len(search_space)

    if search_space_len > 10 and search_space_len <= 100:
        population_size = 4
        cross_over_rate = 0.6

    elif search_space_len > 100 and search_space_len <= 1000:
        population_size = 8
        cross_over_rate = 0.65

    elif search_space_len > 1000 and search_space_len <= 10000:
        population_size = 16
        cross_over_rate = 0.7

    elif search_space_len > 10000 and search_space_len <= 100000:
        population_size = 64
        cross_over_rate = 0.75

    elif search_space_len > 100000 and search_space_len <= 1000000:
        population_size = 128
        cross_over_rate = 0.8
    else:
        population_size = 200
        cross_over_rate = 0.85

    mutation_rate = random.uniform(1/search_space_len, 1/population_size)

    return population_size, cross_over_rate, mutation_rate

def calculate_fittest(population):

    l_individuals_width = []

    for individual in population:
        min_width_box = arrange_circles_and_get_min_width_box(list(individual))
        l_individuals_width.append(min_width_box)

    best_individual_width = min(l_individuals_width)
    best_individual = population[l_individuals_width.index(best_individual_width)]

    return best_individual_width, best_individual

def map_parents(parent1, parent2):

    parent1 = parent1.copy()
    parent2 = parent2.copy()

    indices1 = list(range(0, len(parent1)))
    indices2 = []

    if len(set(parent1)) == len(set(parent2)):

        d_occurence = {}
        for index in range(len(parent1)):
            element_1 = parent1[index]
            d_occurence[element_1] = d_occurence.get(element_1, 0) + 1

            index = [i for i, n in enumerate(parent2) if n == element_1][d_occurence[element_1] - 1]
            indices2.append(index)

    return indices1, indices2

def cycle_crossover(parent1_, parent2_):

    parent1 = parent1_.copy()
    parent2 = parent2_.copy()

    parent1, parent2 = map_parents(parent1, parent2)

    cycles = [-1] * len(parent1)
    cycle_no = 1
    cyclestart = (i for i, v in enumerate(cycles) if v < 0)

    for pos in cyclestart:

        while cycles[pos] < 0:
            cycles[pos] = cycle_no
            pos = parent1.index(parent2[pos])

        cycle_no += 1

    child1 = [parent1_[i] if n % 2 else parent2_[i] for i, n in enumerate(cycles)]
    child2 = [parent2_[i] if n % 2 else parent1_[i] for i, n in enumerate(cycles)]

    return child1, child2

def apply_crossover(curr_population, cross_over_rate):

    curr_population = curr_population.copy()
    consecutive_pairs = zip(curr_population[::2], curr_population[1::2])

    for pair in list(consecutive_pairs):

        parent1 = list(pair[0])
        parent2 = list(pair[1])

        if numpy.random.random() < cross_over_rate:  # if pc is greater than random number
            # child1, child2 = ordered_crossover(parent1, parent2)
            child1, child2 = cycle_crossover(parent1, parent2)
            curr_population.append(child1)
            curr_population.append(child2)

        # copy parents - already in curr_population so, pass and not append again
        else:
            pass

    return curr_population

def swap_mutation(list_, pos1, pos2):

    list_[pos1], list_[pos2] = list_[pos2], list_[pos1]

    return list_

def apply_mutation(curr_population, mutation_rate):

    curr_population = curr_population.copy()

    for index, individual in enumerate(curr_population):

        if numpy.random.random() < mutation_rate:

            i = random.choice(list(range(len(individual))))
            j = random.choice(list(range(len(individual))))
            while i == j:
                j = random.choice(list(range(len(individual))))

            individual = swap_mutation(individual, i, j)
            curr_population[index] = individual

    return curr_population

def apply_roulette_wheel(population):

    l_individuals_width = []
    total_width = 0

    for index, individual in enumerate(population):
        min_width_box = arrange_circles_and_get_min_width_box(list(individual))
        total_width += min_width_box
        l_individuals_width.append([index, individual, min_width_box])

    l_individuals_width.sort(key=lambda x: x[2], reverse=True)
    s_l_individuals_info = [[individuals_width[0],individuals_width[1], individuals_width[2] / total_width] for individuals_width in
                            l_individuals_width]

    prev_prob = 1

    for index, individual_info in enumerate(s_l_individuals_info):

        last_prob = prev_prob
        first_prob = prev_prob - individual_info[2]
        s_l_individuals_info[index].extend([first_prob, last_prob])
        prev_prob = first_prob

        if index == len(s_l_individuals_info) - 1:
            s_l_individuals_info[index][3] = 0

    return s_l_individuals_info

def apply_selection(curr_population, population_size):

    if len(curr_population) != population_size:

        population_with_info = apply_roulette_wheel(curr_population)
        new_population = []

        for i in range(population_size):
            rand_num = random.uniform(0, 1)

            for individual_info in population_with_info:

                first_interval = individual_info[3]
                last_interval = individual_info[4]

                if rand_num >= first_interval and rand_num <= last_interval:
                    new_population.append(individual_info[1])
                    break

    else:
        new_population = curr_population

    return new_population

def get_genetic_algorithm_solution(search_space, iter_num=100):

    population_size, cross_over_rate, mutation_rate = initialize_population_and_other_params(search_space)
    init_population = random.sample(search_space, population_size)
    curr_population = [list(i) for i in init_population]

    best_individual_width, best_individual = calculate_fittest(curr_population)

    for iter in range(iter_num):

        random.shuffle(curr_population)
        curr_population = apply_crossover(curr_population, cross_over_rate)
        curr_population = apply_mutation(curr_population, mutation_rate)
        curr_population = apply_selection(curr_population, population_size)

        best_individual_of_population_width, best_individual_of_population = calculate_fittest(curr_population)

        if best_individual_of_population_width < best_individual_width:
            best_individual_width = best_individual_of_population_width
            best_individual = best_individual_of_population

    return best_individual_width, best_individual

l_results = []
limit_size = [7, 8, 9, 10, 11, 12]
range_complexities = [(1, 60), (1, 60), (1, 60), (1, 60), (1, 60), (1, 60)]

prev_inputs = [[51, 3, 13, 23, 45, 23, 39],
               [50, 17, 50, 45, 5, 45, 43, 2],
               [39, 32, 49, 3, 56, 13, 12, 35, 43],
               [57, 13, 59, 44, 16, 37, 38, 9, 9, 38],
               [48, 3, 6, 39, 1, 48, 17, 13, 60, 52, 16],
               [24, 16, 56, 27, 2, 24, 39, 27, 2, 32, 6, 18]]

for i, (size, comlexity) in enumerate(zip(limit_size, range_complexities)):

    d_result = {}
    print("------------------------------------------")
    ###############################
    start = comlexity[0]; stop = comlexity[1]
    # input_circle_list = [random.randint(start, stop) for iter in range(size)]
    input_circle_list = prev_inputs[i]

    print("\ninput_circles: ", input_circle_list)
    d_result["size"] = size
    d_result["comlexity"] = comlexity
    ###############################
    search_space = get_search_space_domain(input_circle_list)
    d_result["total_search_space_size"] = len(search_space)
    print("search_space_len: ", len(search_space))
    """
        HW-PART2
    """
    start_time = time.time()
    ga_min_width_box, ga_arrangement_circles = get_genetic_algorithm_solution(search_space)
    ga_performance = time.time() - start_time
    print("ga_min_width_box: ", ga_min_width_box, "ga_arrangement_circles: ", ga_arrangement_circles, "ga_performance: ", ga_performance)

    d_result["ga_min_width_box"] = ga_min_width_box
    d_result["ga_arrangement_circles"] = ga_arrangement_circles
    d_result["ga_performance"] = ga_performance

    """
        HW-PART1
    """

    # ############################################################################################################################
    start_time = time.time()
    bf_min_width_box, bf_arrangement_circles = get_brute_force_solution(search_space)
    bf_performance = time.time() - start_time
    print("bf_min_width_box: ", bf_min_width_box, "bf_arrangement_circles: ", bf_arrangement_circles, "bf_performance: ", bf_performance)

    d_result["bf_min_width_box"] = bf_min_width_box
    d_result["bf_arrangement_circles"] = bf_arrangement_circles
    d_result["bf_performance"] = bf_performance
    ############################################################################################################################
    start_time = time.time()
    greedy_min_width_box, greedy_arrangement_circles = get_greedy_solution(input_circle_list)
    greedy_performance = time.time() - start_time
    print("gr_min_width_box: ", greedy_min_width_box, "greedy_arrangement_circles: ", greedy_arrangement_circles, "gr_performance: ", greedy_performance)

    d_result["gr_min_width_box"] = greedy_min_width_box
    d_result["gr_arrangement_circles"] = greedy_arrangement_circles
    d_result["gr_performance"] = greedy_performance
    ############################################################################################################################
    start_time = time.time()
    ils_min_width_box, ils_arrangement_circles = get_iterated_local_search_solution(search_space, iteration_num=100)
    ils_performance = time.time() - start_time
    print("ils_min_width_box: ", ils_min_width_box, "ils_arrangement_circles: ", ils_arrangement_circles, "ils_performance: ", ils_performance)

    d_result["ils_min_width_box"] = ils_min_width_box
    d_result["ils_arrangement_circles"] = ils_arrangement_circles
    d_result["ils_performance"] = ils_performance
    ############################################################################################################################
    start_time = time.time()
    vns_min_width_box, vns_arrangement_circles = get_variable_neighborhood_search_solution(search_space, iteration_num=50, neighborhoods=5)
    vns_performance = time.time() - start_time
    print("vns_min_width_box: ", vns_min_width_box, "vns_arrangement_circles: ", vns_arrangement_circles, "vns_performance: ", vns_performance)

    d_result["vns_min_width_box"] = vns_min_width_box
    d_result["vns_arrangement_circles"] = vns_arrangement_circles
    d_result["vns_performance"] = vns_performance
    ############################################################################################################################
    start_time = time.time()
    sa_min_width_box, sa_arrangement_circles = get_simulated_annealing_solution(search_space, T=1000000, r=0.9, lb_T=0.1, MAX_TRIES=100)
    sa_performance = time.time() - start_time
    print("sa_min_width_box: ", sa_min_width_box, "sa_arrangement_circles: ", sa_arrangement_circles, "sa_performance: ", sa_performance)

    d_result["sa_min_width_box"] = sa_min_width_box
    d_result["sa_arrangement_circles"] = sa_arrangement_circles
    d_result["sa_performance"] = sa_performance
    ############################################################################################################################
    start_time = time.time()
    ts_min_width_box, ts_arrangement_circles = get_tabu_search_solution(search_space)
    ts_performance = time.time() - start_time
    print("ts_min_width_box: ", ts_min_width_box, "ts_arrangement_circles: ", ts_arrangement_circles, "ts_performance: ", ts_performance)

    d_result["ts_min_width_box"] = ts_min_width_box
    d_result["ts_arrangement_circles"] = ts_arrangement_circles
    d_result["ts_performance"] = ts_performance

    l_results.append(d_result)

    with open('l_results3.pickle', 'wb') as b:
        pickle.dump(l_results,b)


"""

OUTPUTS
-------
input_circles:  [51, 3, 13, 23, 45, 23, 39]
search_space_len:  1260
ga_min_width_box:  355.7147324504639 ga_arrangement_circles:  [39, 13, 3, 51, 23, 45, 23] ga_performance:  0.16425728797912598
bf_min_width_box:  354.61204270237045 bf_arrangement_circles:  (39, 23, 51, 13, 45, 23, 3) bf_performance:  0.04737424850463867
gr_min_width_box:  409.83226850080126 greedy_arrangement_circles:  [3, 13, 23, 23, 39, 45, 51] gr_performance:  0.0
ils_min_width_box:  354.61204270237045 ils_arrangement_circles:  [23, 51, 3, 13, 45, 23, 39] ils_performance:  0.30152177810668945
vns_min_width_box:  354.61204270237045 vns_arrangement_circles:  [23, 45, 13, 3, 51, 23, 39] vns_performance:  1.065000057220459
sa_min_width_box:  354.61204270237045 sa_arrangement_circles:  [23, 45, 23, 51, 39, 3, 13] sa_performance:  1.0223500728607178
ts_min_width_box:  354.61204270237045 ts_arrangement_circles:  [23, 45, 13, 51, 23, 39, 3] ts_performance:  1.5017633438110352
------------------------------------------

input_circles:  [50, 17, 50, 45, 5, 45, 43, 2]
search_space_len:  5040
ga_min_width_box:  482.22355265691436 ga_arrangement_circles:  [45, 45, 50, 17, 5, 50, 2, 43] ga_performance:  0.20310330390930176
bf_min_width_box:  482.20082244860544 bf_arrangement_circles:  (45, 50, 2, 17, 50, 5, 43, 45) bf_performance:  0.2052912712097168
gr_min_width_box:  531.6832671700481 greedy_arrangement_circles:  [2, 5, 17, 43, 45, 45, 50, 50] gr_performance:  0.0
ils_min_width_box:  482.20082244860544 ils_arrangement_circles:  [45, 50, 2, 17, 5, 50, 43, 45] ils_performance:  0.35931968688964844
vns_min_width_box:  482.20082244860544 vns_arrangement_circles:  [45, 2, 43, 50, 17, 50, 5, 45] vns_performance:  2.4156136512756348
sa_min_width_box:  482.20082244860544 sa_arrangement_circles:  [17, 5, 50, 50, 2, 45, 45, 43] sa_performance:  1.3283448219299316
ts_min_width_box:  482.20082244860544 ts_arrangement_circles:  [45, 43, 50, 17, 2, 50, 5, 45] ts_performance:  2.793203353881836
------------------------------------------

input_circles:  [39, 32, 49, 3, 56, 13, 12, 35, 43]
search_space_len:  181440
ga_min_width_box:  504.5773118587358 ga_arrangement_circles:  [39, 32, 43, 12, 56, 13, 49, 3, 35] ga_performance:  2.204030990600586
bf_min_width_box:  504.34786538935214 bf_arrangement_circles:  (35, 43, 12, 3, 56, 13, 49, 32, 39) bf_performance:  9.643509149551392
gr_min_width_box:  583.1467273267466 greedy_arrangement_circles:  [3, 12, 13, 32, 35, 39, 43, 49, 56] gr_performance:  0.0
ils_min_width_box:  504.34786538935214 ils_arrangement_circles:  [35, 43, 3, 12, 56, 13, 49, 32, 39] ils_performance:  1.040224313735962
vns_min_width_box:  504.34786538935214 vns_arrangement_circles:  [35, 43, 3, 12, 56, 13, 49, 32, 39] vns_performance:  6.473365068435669
sa_min_width_box:  504.585591266374 sa_arrangement_circles:  [13, 39, 43, 12, 3, 35, 49, 32, 56] sa_performance:  1.3211064338684082
ts_min_width_box:  504.3478653893522 ts_arrangement_circles:  [39, 32, 49, 13, 56, 12, 43, 35, 3] ts_performance:  3.979680299758911
------------------------------------------

input_circles:  [57, 13, 59, 44, 16, 37, 38, 9, 9, 38]
search_space_len:  453600
ga_min_width_box:  552.9193753845507 ga_arrangement_circles:  [38, 9, 59, 16, 57, 13, 38, 44, 9, 37] ga_performance:  2.5634982585906982
bf_min_width_box:  549.6916403977198 bf_arrangement_circles:  (38, 9, 37, 9, 57, 16, 59, 13, 44, 38) bf_performance:  29.32717752456665
gr_min_width_box:  637.9690993932993 greedy_arrangement_circles:  [9, 9, 13, 16, 37, 38, 38, 44, 57, 59] gr_performance:  0.0
ils_min_width_box:  549.6916403977198 ils_arrangement_circles:  [38, 9, 44, 13, 59, 16, 57, 9, 37, 38] ils_performance:  1.617713212966919
vns_min_width_box:  549.6916403977198 vns_arrangement_circles:  [38, 44, 13, 59, 16, 57, 9, 37, 9, 38] vns_performance:  13.93344783782959
sa_min_width_box:  549.6983073607091 sa_arrangement_circles:  [13, 16, 59, 38, 37, 57, 44, 9, 9, 38] sa_performance:  1.4996461868286133
ts_min_width_box:  549.6983073607091 ts_arrangement_circles:  [38, 38, 9, 44, 13, 59, 16, 57, 9, 37] ts_performance:  6.051348686218262
------------------------------------------

input_circles:  [48, 3, 6, 39, 1, 48, 17, 13, 60, 52, 16]
search_space_len:  9979200
ga_min_width_box:  517.3180924607128 ga_arrangement_circles:  [39, 48, 16, 52, 17, 60, 13, 48, 3, 1, 6] ga_performance:  6.174573183059692
bf_min_width_box:  516.1560736640935 bf_arrangement_circles:  (48, 1, 3, 17, 60, 16, 52, 13, 48, 39, 6) bf_performance:  798.5788938999176
gr_min_width_box:  641.6149182398522 greedy_arrangement_circles:  [1, 3, 6, 13, 16, 17, 39, 48, 48, 52, 60] gr_performance:  0.0
ils_min_width_box:  516.1560736640935 ils_arrangement_circles:  [39, 6, 48, 17, 60, 16, 52, 13, 1, 3, 48] ils_performance:  2.421302080154419
vns_min_width_box:  516.1560736640935 vns_arrangement_circles:  [48, 17, 60, 16, 52, 3, 13, 48, 1, 39, 6] vns_performance:  34.992786169052124
sa_min_width_box:  516.1560736640936 sa_arrangement_circles:  [1, 17, 52, 48, 13, 6, 39, 48, 16, 60, 3] sa_performance:  1.749929666519165
ts_min_width_box:  516.1560736640936 ts_arrangement_circles:  [48, 13, 3, 52, 16, 1, 60, 17, 6, 48, 39] ts_performance:  9.298790454864502
------------------------------------------

input_circles:  [24, 16, 56, 27, 2, 24, 39, 27, 2, 32, 6, 18]
search_space_len:  29937600
ga_min_width_box:  495.6378357186656 ga_arrangement_circles:  [24, 6, 32, 27, 2, 27, 24, 2, 39, 16, 56, 18] ga_performance:  10.10781216621399
bf_min_width_box:  493.76177013730074 bf_arrangement_circles:  (27, 24, 2, 39, 2, 16, 56, 18, 32, 6, 24, 27) bf_performance:  2994.715124130249
gr_min_width_box:  572.6885561298654 greedy_arrangement_circles:  [2, 2, 6, 16, 18, 24, 24, 27, 27, 32, 39, 56] gr_performance:  0.0
ils_min_width_box:  493.76177013730074 ils_arrangement_circles:  [27, 24, 39, 16, 56, 18, 2, 32, 24, 6, 2, 27] ils_performance:  3.368675708770752
vns_min_width_box:  493.76177013730074 vns_arrangement_circles:  [27, 24, 2, 32, 18, 56, 6, 16, 39, 24, 2, 27] vns_performance:  84.73001980781555
sa_min_width_box:  494.4093133286119 sa_arrangement_circles:  [27, 27, 6, 56, 18, 24, 2, 2, 24, 16, 32, 39] sa_performance:  1.9977481365203857
ts_min_width_box:  493.8500818918693 ts_arrangement_circles:  [24, 2, 39, 16, 56, 6, 18, 32, 24, 2, 27, 27] ts_performance:  13.629377126693726

Process finished with exit code 0

"""
