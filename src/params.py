import json
from multiprocessing import Pool
from mealpy import FloatVar
import random

import numpy as np

default_params = {
    'esm_30': {
        'l1_dim': 600,
        'l2_dim': 300,
        'dropout_rate': 0.5,
        'leakyrelu_1_alpha': 0.1
    },
    'esm_33': {
        'l1_dim': 600,
        'l2_dim': 300,
        'dropout_rate': 0.5,
        'leakyrelu_1_alpha': 0.1
    },
    'taxa': {
        'l1_dim': 128,
        'l2_dim': 64,
        'dropout_rate': 0.5,
        'leakyrelu_1_alpha': 0.1
    },
    'taxa_profile': {
        'l1_dim': 128,
        'l2_dim': 64,
        'dropout_rate': 0.5,
        'leakyrelu_1_alpha': 0.1
    },
    'final': {
        'dropout_rate': 0.5,
        'final_dim': 256,
        'patience': 10,
        'epochs': 40,
        'learning_rate': 0.0003,
        'batch_size': 256
    }
}

param_bounds = {
    'esm_30': {
        'l1_dim': [600,600],
        'l2_dim': [300,300],
        'dropout_rate': [0.5, 0.5],
        'leakyrelu_1_alpha': [0.1, 0.1]
    },
    'esm_33': {
        'l1_dim': [600,600],
        'l2_dim': [300,300],
        'dropout_rate': [0.5, 0.5],
        'leakyrelu_1_alpha': [0.1, 0.1]
    },
    'taxa': {
        'l1_dim': [128, 128],
        'l2_dim': [64, 64],
        'dropout_rate': [0.5, 0.5],
        'leakyrelu_1_alpha': [0.1, 0.1],
    },
    'taxa_profile': {
        'l1_dim': [64, 300],
        'l2_dim': [32, 160],
        'dropout_rate': [0.3, 0.8],
        'leakyrelu_1_alpha': [0.05, 0.2],
    },
    'final': {
        'dropout_rate': [0.5,0.5],
        'final_dim': [256,256],
        'patience': [10,10],
        'epochs': [40,40],
        'learning_rate': [0.0003,0.0003],
        'batch_size': [256,256]
    }
}

param_bounds2 = {
    'esm_30': {
        'l1_dim': [400,700],
        'l2_dim': [150,400],
        'dropout_rate': [0.4, 0.6],
        'leakyrelu_1_alpha': [0.05, 0.15]
    },
    'esm_33': {
        'l1_dim': [400,700],
        'l2_dim': [150,400],
        'dropout_rate': [0.4, 0.6],
        'leakyrelu_1_alpha': [0.05, 0.15]
    },
    'taxa': {
        'l1_dim': [128, 128],
        'l2_dim': [64, 64],
        'dropout_rate': [0.5, 0.5],
        'leakyrelu_1_alpha': [0.1, 0.1],
    },
    'taxa_profile': {
        'l1_dim': [150, 250],
        'l2_dim': [100, 140],
        'dropout_rate': [0.4, 0.6],
        'leakyrelu_1_alpha': [0.04, 0.1],
    },
    'final': {
        'dropout_rate': [0.4,0.6],
        'final_dim': [200,300],
        'patience': [8,12],
        'epochs': [35,45],
        'learning_rate': [0.0002,0.0006],
        'batch_size': [200,400]
    }
}

param_types = {
    'l1_dim': int,
    'l2_dim': int,
    'leakyrelu_1_alpha': float,
    'dropout_rate': float,
    'final_dim': int,
    'patience': int,
    'epochs': int,
    'learning_rate': float,
    'batch_size': int
}

class ProblemTranslator:
    def __init__(self) -> None:
        self.params_list = []
        self.upper_bounds = []
        self.lower_bounds = []
        param_groups = sorted(param_bounds2.keys())
        for key in param_groups:
            param_names = sorted(param_bounds2[key].keys())
            for name in param_names:
                lower = float(param_bounds2[key][name][0])
                upper = float(param_bounds2[key][name][1])
                self.lower_bounds.append(lower)
                self.upper_bounds.append(upper)
                self.params_list.append((key, name))

    def to_bounds(self):
        return FloatVar(lb=self.lower_bounds, ub=self.upper_bounds)
    
    def decode(self, vec):
        new_param_dict = {}
        for first, second in self.params_list:
            new_param_dict[first] = {}
        for i in range(len(vec)):
            first, second = self.params_list[i]
            converter = param_types[second]
            val = converter(vec[i])
            new_param_dict[first][second] = val
        
        return new_param_dict

TRANSLATOR = ProblemTranslator()

class RandomSearchMetaheuristic:
    def __init__(self, pop_size, upper_bounds, lower_bounds, n_jobs = 24) -> None:
        assert len(upper_bounds) == len(lower_bounds)
        self.pop_size = pop_size
        self.bounds = [(lower_bounds[i], upper_bounds[i]) for i in range(len(upper_bounds))]
        self.n_jobs = n_jobs
        random.seed(1337)
        self.generate_population()
    
    def generate_population(self):
        self.population = []
        for _ in range(self.pop_size):
            new_solution = []
            for lb, ub in self.bounds:
                val = random.uniform(lb, ub)
                new_solution.append(val)
            self.population.append(new_solution)
    
    def run_tests(self, objective_func):

        with Pool(self.n_jobs) as pool:
            fitness_vec = pool.map(objective_func, self.population)
        
        solutions_with_fitness = [(self.population[i], fitness_vec[i])
            for i in range(self.pop_size)]
        
        solutions_with_fitness.sort(key = lambda tp: tp[1])

        best_solution, best_fitness = solutions_with_fitness[-1]
        n_top = int(len(self.population) / 2)
        report = []
        top_best = solutions_with_fitness[-n_top:]
        for s, f in top_best:
            report.append('Top ' + str(n_top) + ' solutions:')
            solution_str = json.dumps(TRANSLATOR.decode(s), indent=4)
            report += solution_str.split('\n')
            report.append('Mean ROC AUC: ' + str(f))
        
        for i in range(len(TRANSLATOR.params_list)):
            param_name = TRANSLATOR.params_list[i]
            param_values = [s[i] for s, f in top_best]
            std = np.std(param_values)
            mean = np.mean(param_values)
            report.append(str(param_name) + ' mean: ' + str(mean) + '; std:' + str(std))
        report = '\n'.join(report)
        print(report)
        return best_solution, best_fitness, report



