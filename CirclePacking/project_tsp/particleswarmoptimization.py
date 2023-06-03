# ref : http://www.theprojectspot.com/tutorial-post/ant-colony-optimization-for-hackers/10

import random
import math
from matplotlib import pyplot as plt
import time
import datetime

class Edge:
    def __init__(self, a, b, weight):
        self.a = a
        self.b = b
        self.weight = weight

class Particle:

    def __init__(self, num_nodes, edges, pos_vector):
        self.num_nodes = num_nodes
        self.edges = edges
        self.pos_vector = pos_vector
        self.best_experience_path = pos_vector
        self.velocity = math.ceil(self.num_nodes / 2)
        self.vel_weight = math.ceil(self.velocity / 2)
        self.velocity_vector = self.set_initial_velocity()
        self.c1 = math.ceil(self.velocity / 2)
        self.r1 = math.ceil(self.velocity / 2)
        self.c2 = math.ceil(self.velocity / 2)
        self.r2 = math.ceil(self.velocity / 2)
        self.best_experience_distance = self.find_fitness_function_value(self)

    def find_fitness_function_value(self, particle):
        self.distance = 0.0
        for i in range(self.num_nodes):
            source=particle.pos_vector[i]; destination = particle.pos_vector[(i + 1) % self.num_nodes]
            self.distance += self.edges[source][destination].weight
        return self.distance

    def set_initial_velocity(self):

        vel_vec = []
        for i in range(self.velocity):
            first_pos = random.choice(list(range(self.num_nodes)))
            second_pos = random.choice(list(range(self.num_nodes)))
            while first_pos == second_pos:
                second_pos = random.choice(self.pos_vector)
            vel_vec.append((first_pos, second_pos))

        return vel_vec

    def swap_operation(self, vector, index1, index2):

        vector[index1], vector[index2] = vector[index2], vector[index1]


    def get_velocity_with_positions(self, pos_vec1, pos_vec2):

        pos_vec1 = pos_vec1.copy()
        pos_vec2 = pos_vec2.copy()
        new_velocity = []
        for index_1, pos in enumerate(pos_vec1):
            index_2 = pos_vec2.index(pos)
            if index_1 != index_2:
                new_velocity.append((index_1, index_2))
                self.swap_operation(pos_vec2, index_1, index_2)

        return new_velocity

    def update_velocity(self, particle):

        velocity_with_best_experience = self.get_velocity_with_positions(particle.best_experience_path, particle.pos_vector)
        velocity_with_global_best = self.get_velocity_with_positions(pso.global_best_path, particle.pos_vector)

        velocity_part = []; experience_part = []; global_part = []
        if len(particle.velocity_vector)>0:
            velocity_part = particle.velocity_vector[0:math.ceil(particle.vel_weight)]
        if len(velocity_with_best_experience)>0:
            experience_part = velocity_with_best_experience[0:math.ceil(particle.c1 * particle.r1)]
        if len(velocity_with_global_best)>0:
            global_part = velocity_with_global_best[0:math.ceil(particle.c2 * particle.r2)]

        new_velocity = list(set(velocity_part + experience_part + global_part))
        particle.velocity_vector = new_velocity

        return new_velocity

    def update_position_vector(self, particle, new_velocity):

        for swaps in new_velocity:
            self.swap_operation(particle.pos_vector, swaps[0], swaps[1])

    def update_local_best_position(self, particle, distance_path):

        if distance_path < particle.best_experience_distance:
            particle.best_experience_path = particle.pos_vector
            particle.best_experience_distance = distance_path

class ParticleSwarmOptimization:

    def __init__(self, particle_size=5, iter_num=500, nodes=None, labels=None):

        self.particle_size = particle_size
        self.n_iter = iter_num
        self.node_num = len(nodes)
        self.nodes = nodes

        if labels is not None:
            self.labels = labels
        else:
            self.labels = range(1, self.node_num + 1)

        self.edges = [[None] * self.node_num for _ in range(self.node_num)]

        for i in range(self.node_num):
            for j in range(i + 1, self.node_num):
                self.edges[i][j] = self.edges[j][i] = Edge(i, j, math.sqrt(pow(self.nodes[i][0] - self.nodes[j][0], 2.0) + pow(self.nodes[i][1] - self.nodes[j][1], 2.0)))

        self.particle_solutions = []
        self.initialize_population()
        self.particles = [Particle(self.node_num, self.edges, self.particle_solutions[i]) for i in range(self.particle_size)]

        self.global_best_path = None
        self.global_best_distance = float("inf")
        for i in range(self.particle_size):
            if self.particles[i].best_experience_distance < self.global_best_distance:
                self.global_best_path = self.particles[i].best_experience_path
                self.global_best_distance = self.particles[i].best_experience_distance

    def initialize_population(self):

        for i in range(self.particle_size):
            solution = list(range(self.node_num))
            random.shuffle(solution)
            while self.particle_solutions.count(solution) != 0:
                random.shuffle(solution)
            self.particle_solutions.append(solution)

    def particle_swarm_solution(self):
        for t in range(self.n_iter):
            for particle in self.particles:
                new_velocity = particle.update_velocity(particle)
                particle.update_position_vector(particle, new_velocity)
                distance_path = particle.find_fitness_function_value(particle)
                particle.update_local_best_position(particle, distance_path)
                if particle.best_experience_distance < self.global_best_distance:
                    self.global_best_path = particle.best_experience_path
                    self.global_best_distance = particle.distance
                # else:
                #     particle.vel_weight = particle.vel_weight * 1.1

    def run(self):
        print('Started Particle Swarm Optimization')
        self.particle_swarm_solution()
        print('Ended Particle Swarm Optimization')
        print('Sequence : <- {0} ->'.format(' - '.join(str(self.labels[i]) for i in self.global_best_path)))
        print('Total distance travelled to complete the tour : {0}\n'.format(round(self.global_best_distance, 2)))

    def plot(self, line_width=1, point_radius=math.sqrt(3.0), annotation_size=10, dpi=150, save=True, name=None):
        x = [self.nodes[i][0] for i in self.global_best_path]
        x.append(x[0])
        y = [self.nodes[i][1] for i in self.global_best_path]
        y.append(y[0])
        plt.plot(x, y, linewidth=line_width, color="lime")
        plt.scatter(x, y, s=math.pi * (point_radius ** 3.0))
        plt.title("PSO")
        for i in self.global_best_path:
            plt.annotate(self.labels[i], self.nodes[i], size=annotation_size, color="red")
        if save:
            if name is None:
                name = 'pso.png'
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

    # nodes = [(2, 1), (5, 5), (9, 3), (12, 4), (9, 0)]
    nodes = read_city_data()
    particle_size = len(nodes) * 10
    iter_num = len(nodes) * 10
    print("# nodes: ", len(nodes), " particle_size: ", particle_size, " iter_num: ", iter_num)
    pso = ParticleSwarmOptimization(particle_size=particle_size, iter_num=iter_num, nodes=nodes)
    start_time = time.time()
    pso.run()
    end_time = time.time()
    print("exe_time_sec", 1000*(end_time-start_time) )
    pso.plot()
    """
                    Min_dist    time_ms
    T-GA            1145.76     10218
    ACP-GA          628.96      9501
    IS-GA           513.05      15898
    RCDM-GA         466.84      13462
    proposed PSO1   1131.64     4641.80    # particle_size = 102  iter_num = 510
    proposed PSO2   1072.81     24334.82       # particle_size = 510  iter_num = 510
    """
