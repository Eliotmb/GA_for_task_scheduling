import random
from deap import base, creator, tools, algorithms

# --- PROBLEM SETUP ---
# Tasks: (ID, Required Skill, Estimated Hours, Dependencies)
tasks = [
    (0, "frontend", 20, []),  # T1
    (1, "frontend", 15, [0]),  # T2 (depends on T1)
    (2, "backend", 30, []),  # T3
    (3, "backend", 25, [2]),  # T4 (depends on T3)
    (4, "full-stack", 10, [0, 2]),  # T5 (depends on T1 and T3)
    (5, "QA", 8, [1, 3]),  # T6 (depends on T2 and T4)
    (6, "frontend", 12, []),  # T7
    (7, "backend", 18, []),  # T8
    (8, "QA", 10, [6, 7]),  # T9 (depends on T7 and T8)
    (9, "full-stack", 20, [4, 5]),  # T10 (depends on T5 and T6)
]

# Developers: (ID, Skills, Weekly Capacity (hours))
developers = [
    (0, ["frontend"], 40),  # Dev1
    (1, ["backend"], 35),  # Dev2
    (2, ["full-stack", "frontend", "backend"], 45),  # Dev3
    (3, ["QA"], 30),  # Dev4
]

# --- GENETIC ALGORITHM SETUP ---
# Fitness function: Minimize penalties (skill mismatch + overwork)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


# Gene: Assign a developer to a task (randomly initialized)
def assign_task():
    return random.randint(0, len(developers) - 1)


# Create individual and population
toolbox.register("gene", assign_task)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.gene, n=len(tasks))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# --- FITNESS FUNCTION ---
def evaluate(individual):
    total_penalty = 0
    dev_hours = {dev_id: 0 for dev_id in range(len(developers))}

    # Check each task assignment
    for task_idx, dev_id in enumerate(individual):
        task = tasks[task_idx]
        dev = developers[dev_id]

        # Penalty 1: Developer lacks required skill
        if task[1] not in dev[1]:
            total_penalty += 100  # Heavy penalty for skill mismatch

        # Penalty 2: Check dependencies (e.g., if dependent tasks are assigned to the same dev)
        for dep in task[3]:
            if individual[dep] == dev_id:
                total_penalty += 50  # Context-switching penalty

        # Track developer hours
        dev_hours[dev_id] += task[2]

    # Penalty 3: Overworked developers
    for dev_id, hours in dev_hours.items():
        if hours > developers[dev_id][2]:
            total_penalty += (hours - developers[dev_id][2]) * 10  # Linear penalty

    return (total_penalty,)


toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=len(developers) - 1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


# --- RUN GA ---
def main():
    population = toolbox.population(n=50)
    NGEN = 40
    CXPB = 0.7
    MUTPB = 0.2

    # Evolutionary loop
    algorithms.eaSimple(
        population,
        toolbox,
        cxpb=CXPB,
        mutpb=MUTPB,
        ngen=NGEN,
        verbose=True
    )

    # Best solution
    best_ind = tools.selBest(population, k=1)[0]
    print("\nBest Schedule:")
    for task_idx, dev_id in enumerate(best_ind):
        print(f"Task {task_idx} (needs {tasks[task_idx][1]}): Assigned to Developer {dev_id}")


if __name__ == "__main__":
    main()
