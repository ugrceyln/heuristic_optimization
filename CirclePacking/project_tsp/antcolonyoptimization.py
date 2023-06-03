# ref : http://www.theprojectspot.com/tutorial-post/ant-colony-optimization-for-hackers/10

import random
import math
from matplotlib import pyplot as plt
import time
import datetime

class Edge:
    def __init__(self, a, b, weight, initial_pheromone):
        self.a = a
        self.b = b
        self.weight = weight
        self.pheromone = initial_pheromone

class Ant:
    def __init__(self, alpha, beta, num_nodes, edges):
        self.alpha = alpha
        self.beta = beta
        self.num_nodes = num_nodes
        self.edges = edges
        self.tour = None
        self.distance = 0.0

    def select_node(self):
        roulette_wheel = 0.0
        unvisited_nodes = [node for node in range(self.num_nodes) if node not in self.tour]

        for unvisited_node in unvisited_nodes:
            roulette_wheel += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                              pow((1 / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)

        random_value = random.uniform(0.0, roulette_wheel)
        wheel_position = 0.0
        for unvisited_node in unvisited_nodes:
            wheel_position += pow(self.edges[self.tour[-1]][unvisited_node].pheromone, self.alpha) * \
                              pow((1 / self.edges[self.tour[-1]][unvisited_node].weight), self.beta)

            if wheel_position >= random_value:
                return unvisited_node

    def find_tour(self):
        self.tour = [random.randint(0, self.num_nodes - 1)]
        while len(self.tour) < self.num_nodes:
            unvisited_node = self.select_node()
            self.tour.append(unvisited_node)
        return self.tour

    def get_distance(self):
        self.distance = 0.0
        for i in range(self.num_nodes):
            self.distance += self.edges[self.tour[i]][self.tour[(i + 1) % self.num_nodes]].weight
        return self.distance

class AntColonyOptimization:

    def __init__(self, ant_size=5, t_max=300, alpha=1.0, beta=3.0, init_pheromone=1.0, Q=10.0, rho=0.1, nodes=None, labels=None):

        self.ant_size = ant_size
        self.rho = rho
        self.Q = Q
        self.t_max = t_max
        self.node_num = len(nodes)
        self.nodes = nodes

        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.node_num + 1)

        self.edges = [[None] * self.node_num for _ in range(self.node_num)]

        for i in range(self.node_num):
            for j in range(i + 1, self.node_num):
                self.edges[i][j] = self.edges[j][i] = Edge(i, j,
                                                           math.sqrt(pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)),
                                                           init_pheromone)

        self.ants = [Ant(alpha, beta, self.node_num, self.edges) for _ in range(self.ant_size)]
        self.global_best_tour = None
        self.global_best_distance = float("inf")

    def increase_pheromone_on_traversed_path(self, tour, length_of_tour):
        pheromone_to_add = self.Q / length_of_tour
        for i in range(self.node_num):
            self.edges[tour[i]][tour[(i + 1) % self.node_num]].pheromone += pheromone_to_add

    def evaporate_trails(self):

        for i in range(self.node_num):
            for j in range(i + 1, self.node_num):
                self.edges[i][j].pheromone *= (1.0 - self.rho)

    def update_alpha_and_beta(self, ant):
        ant.alpha *= 0.9999
        ant.beta *= 0.9999

    def ant_colony_solution(self):
        for t in range(self.t_max):
            for ant in self.ants:
                tour = ant.find_tour()
                length_of_tour = ant.get_distance()
                self.increase_pheromone_on_traversed_path(tour, length_of_tour)
                if ant.distance < self.global_best_distance:
                    self.global_best_tour = ant.tour
                    self.global_best_distance = ant.distance

                # self.update_alpha_and_beta(ant)

            self.evaporate_trails()

    def run(self):
        print('Started Ant Colony Optimization')
        self.ant_colony_solution()
        print('Ended Ant Colony Optimization')
        print('Sequence : <- {0} ->'.format(' - '.join(str(self.labels[i]) for i in self.global_best_tour)))
        print('Total distance travelled to complete the tour : {0}\n'.format(round(self.global_best_distance, 2)))

    def plot(self, line_width=1, point_radius=math.sqrt(3.0), annotation_size=10, dpi=150, save=True, name=None):
        x = [self.nodes[i][0] for i in self.global_best_tour]
        x.append(x[0])
        y = [self.nodes[i][1] for i in self.global_best_tour]
        y.append(y[0])
        plt.plot(x, y, linewidth=line_width, color="lime")
        plt.scatter(x, y, s=math.pi * (point_radius ** 3.0))
        plt.title("ACO")
        for i in self.global_best_tour:
            plt.annotate(self.labels[i], self.nodes[i], size=annotation_size, color="red")
        if save:
            if name is None:
                name = 'aco.png'
            plt.savefig(name, dpi=dpi)
        plt.show()
        plt.gcf().clear()

def read_city_data():

    cities_coordinates = []

    with open("cities_data") as f:
        for line in f:
            line = line.split()[0].split(",")
            cities_coordinates.append((int(line[0]), int(line[1])))

    return cities_coordinates

if __name__ == '__main__':

    # nodes = [(1, 1), (4, 5), (8, 8), (2, 16), (1, 16)]
    nodes = read_city_data()
    ant_size = 5   # len(nodes)
    t_max = 500     # len(nodes) * 10
    print("# nodes: ", len(nodes), " ant_size: ", ant_size, " t_max: ", t_max)
    acs = AntColonyOptimization(ant_size=ant_size, t_max=t_max, nodes=nodes)
    start_time = time.time()
    acs.run()
    end_time = time.time()
    print("exe_time_sec", 1000*(end_time-start_time) )
    acs.plot()
    """
                    Min_dist    time_ms
    T-GA            1145.76     10218
    ACP-GA          628.96      9501
    IS-GA           513.05      15898
    RCDM-GA         466.84      13462
    proposed ACO1   478.45      5140    # ant_size = 5  t_max = 500
    proposed ACO2   462.53      54137   # ant_size = 51 t_max = 510
    """
