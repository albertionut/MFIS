import random
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns


class State(Enum):
    AVAILABLE = 0
    BORROWED = 1
    RESERVED = 2
    BORROWED_RESERVED = 3

class Guard:
    def __init__(self, a, op, b):
        self.a = a
        self.op = op
        self.b = b

    def eval(self):
        return eval(f"{self.a} {self.op} {self.b}")

    def out(self):
        return f"{self.a} {self.op} {self.b}"

class BookFSM:
    def __init__(self):
        self.state = State.AVAILABLE
        self.BorID = 0
        self.ResID = 0
        self.BrID = 0

    def getBorCustId(self):
        return self.BorID

    def getResCustId(self):
        return self.ResID

    def getBrCustId(self):
        return self.BrID

    def bor(self, s1, x, s2):
        if s1 == State.AVAILABLE:
            if s2 == State.BORROWED:
                self.BorID = x
                return [Guard(x, '>', 0)]
            elif s2 == State.AVAILABLE:
                return [Guard(x, '<=', 0)]

        elif s1 == State.RESERVED:
            if s2 == State.BORROWED:
                self.BorID = x
                return [Guard(x, '==', self.getResCustId())]
            elif s2 == State.RESERVED:
                return [Guard(x, '!=', self.getResCustId())]

        return None

    def res(self, s1, x, s2):
        if s1 == State.AVAILABLE:
            if s2 == State.RESERVED:
                self.ResID = x
                return [Guard(x, '>', 0)]
            elif s2 == State.AVAILABLE:
                return [Guard(x, '<', 0)]

        elif s1 == State.BORROWED:
            if s2 == State.BORROWED_RESERVED:
                self.BrID = x
                return [Guard(x, '>', 0), Guard(x, '!=', self.getBorCustId())]
            elif s2 == State.BORROWED:
                return [Guard(x, '<=', 0), Guard(x, '=', self.getBorCustId())]

        return None

    def ret(self, s1, x, s2):
        if s1 == State.BORROWED:
            if s2 == State.AVAILABLE:
                return [Guard(x, '==', self.getBorCustId())]
            elif s2 == State.BORROWED:
                return [Guard(x, '!=', self.getBorCustId())]

        elif s1 == State.BORROWED_RESERVED:
            if s2 == State.RESERVED:
                self.ResID = x
                return [Guard(x, '==', self.getBorCustId())]
            elif s2 == State.BORROWED_RESERVED:
                return [Guard(x, '!=', self.getBorCustId())]

        return None

    def can(self, s1, x, s2):
        if s1 == State.BORROWED_RESERVED:
            if s2 == State.BORROWED:
                self.BorID = x
                return [Guard(x, '==', self.getBrCustId())]
            elif s2 == State.BORROWED_RESERVED:
                return [Guard(x, '!=', self.getBrCustId())]

        elif s1 == State.RESERVED:
            if s2 == State.AVAILABLE:
                return [Guard(x, '==', self.getBrCustId())]
            elif s2 == State.RESERVED:
                return [Guard(x, '!=', self.getBrCustId())]

        return None


def get_guards(chromosome):
    x1, x2, x3, x4, x5, x6, x7 = chromosome[0], chromosome[1], chromosome[2], \
        chromosome[3], chromosome[4], chromosome[5], chromosome[6]

    guards = [book.bor(A, x1, B)[0],
              book.ret(B, x2, A)[0],
              book.res(A, x3, R)[0],
              book.can(R, x4, A)[0],
              book.bor(A, x5, B)[0]]

    return guards

def obj(guard):
    K = 1
    a, op, b = guard.a, guard.op, guard.b
    if op == '==':
        if abs(a - b) == 0:
            return 0
        else:
            return abs(a - b) + K
    elif op == '!=':
        if abs(a - b) != 0:
            return 0
        else:
            return K
    elif op == '<':
        if a - b < 0:
            return 0
        else:
            return (a - b) + K
    elif op == '<=':
        if a - b <= 0:
            return 0
        else:
            return (a - b) + K
    elif op == '>':
        if b - a < 0:
            return 0
        else:
            return (b - a) + K
    elif op == '>=':
        if b - a <= 0:
            return 0
        else:
            return (b - a) + K


def norm(x):
    return 1 - 1.005 ** (-x)

def fitness(chromosome):
    guards = get_guards(chromosome)
    m = len(guards)
    ap_level = m - 1
    for g in guards:
        if not g.eval():
            return ap_level + norm(obj(g))
        else:
            ap_level = ap_level - 1
    return 0


def selection(population, fitness_scores, tournament_size):
    selected_individuals = []
    while len(selected_individuals) < len(population):
        tournament_indices = random.sample(range(len(population)), tournament_size)
        min_fitness_index = min(tournament_indices, key=lambda i: fitness_scores[i])
        selected_individuals.append(population[min_fitness_index])
    return selected_individuals


def heuristic_crossover(parent1, parent2):
    child = []
    l = 0.2
    for xi, yi in zip(parent1, parent2):
        zi = int(l * (xi - yi) + xi)
        child.append(zi)

    return child

def crossover(selected_individuals, fitness_scores):
    i1 = random.randint(0, 19)
    i2 = i1
    while i2 == i1:
        i2 = random.randint(0, 19)
    if fitness_scores[i1] > fitness_scores[i2]:
        child = heuristic_crossover(selected_individuals[i1], selected_individuals[i2])
    else:
        child = heuristic_crossover(selected_individuals[i2], selected_individuals[i1])
    return child

# Mutation operator: Random mutation
def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.randint(-1000, 1000)
    return chromosome

# Genetic algorithm to generate test suites
def genetic_algo():

    population_size = 20
    num_generations = 200
    limitA, limitB = -1000, 1000
    crossover_rate = 0.8
    mutation_rate = 0.06
    tournament_size = 5

    fitness_scores = []
    population = []
    chromosome_size = 7

    for _ in range(population_size):
        chromosome = [random.randint(limitA, limitB) for _ in range(chromosome_size)]
        population.append(chromosome)

    # Evolve test suites over multiple generations
    for generation in range(num_generations):
        # Evaluate fitness of each test suite in the population
        fitness_scores = [fitness(chromosome) for chromosome in population]

        # Select individuals for reproduction using tournament selection
        selected_individuals = selection(population, fitness_scores, tournament_size)

        # Perform crossover to create offspring
        offspring = []
        for i in range(0, population_size):
            if random.random() < crossover_rate:
                child = crossover(selected_individuals, fitness_scores)
                offspring.append(child)
            else:
                offspring.append(selected_individuals[i])

        # Mutate offspring
        for i in range(0, population_size):
            offspring[i] = mutate(offspring[i], mutation_rate)

        # Replace population with offspring
        population = offspring

    return population, fitness_scores

if __name__ == "__main__":

    book = BookFSM()
    A = State.AVAILABLE
    B = State.BORROWED
    R = State.RESERVED
    BR = State.BORROWED_RESERVED

    population, fitness_scores = genetic_algo()
    print(population)
    print(fitness_scores)

    # Read fitness and plot result for target paths
    file = open('fitness.txt', 'r')
    fs = []
    for line in file.readlines():
        fs.append([float(f) for f in line.split(',')])

    print(fs)

    # Plot fitness scores for each target path
    plt.figure(figsize=(10, 6))
    for i, scores in enumerate(fs, start=1):
        plt.scatter([i] * len(scores), scores, label=f"Target Path {i}")

    plt.xlabel("Target Path")
    plt.ylabel("Fitness Scores")
    plt.title("Fitness Scores for Each Target Path")
    plt.xticks(range(1, len(fs) + 1), [i for i in range(1, len(fs) + 1)])
    plt.grid(True)
    plt.show()

    # Plotting each list of fitness scores
    num_plots = len(fs)
    rows = num_plots // 2 if num_plots % 2 == 0 else num_plots // 2 + 1

    # Convert fs to a format suitable for seaborn
    data = {'Population': [], 'Fitness Score': []}
    for i, scores in enumerate(fs, start=1):
        for score in scores:
            data['Population'].append(i)
            data['Fitness Score'].append(score)

    all_fitness_scores = [score for sublist in fs for score in sublist]

    # Swarm Plot
    plt.figure(figsize=(10, 6))
    sns.swarmplot(y=all_fitness_scores, color="black")
    plt.ylabel("Fitness Score")
    plt.title("Distribution of Fitness Scores across All Target Paths (Swarm Plot)")
    plt.grid(True)
    plt.show()

    # Get fitness for all 10 target paths
    # file = open('fitness1.txt', 'w')
    # output = ""
    # for f in fitness_scores:
    #     output = output + str(f) + ','
    #
    # output = output[:-1]
    # file.write(output)
    #
    # file = open('fitness1.txt', 'r')
    # text = [float(f) for f in file.read().split(',')]
    # print(text)