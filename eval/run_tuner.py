


MIN_PARAM_VAL = 1.0e-4
MAX_PARAM_VAL = 1.0e-0



# import numpy as np
# from skopt import Optimizer
# from skopt.space import Real
# import time

# # Define the parameter search space (log-uniform for large-scale differences)
# search_space = [Real(MIN_PARAM_VAL, MAX_PARAM_VAL, prior="log-uniform", name="param")]

# # Initialize Bayesian Optimizer
# optimizer = Optimizer(dimensions=search_space, base_estimator="gp", acq_func="EI", random_state=42)

# # Number of evaluations
# n_calls = 10
# evaluations = []

# for i in range(n_calls):
#     # Suggest a new parameter to evaluate
#     next_x = optimizer.ask()[0]
    
#     # Prompt user to enter the cost (or replace with actual function)
#     print(f"Iteration {i+1}/{n_calls}: Suggested parameter: {next_x:.8f}")
    
#     # Here, you should replace this input() call with an actual function evaluation
#     cost = float(input(f"Enter cost for parameter {next_x:.8f}: "))  # User enters cost manually
    
#     # Store the result
#     evaluations.append((next_x, cost))
    
#     # Tell the optimizer the result
#     optimizer.tell([next_x], [cost])

#     print(f"Updated model with (param={next_x:.8f}, cost={cost:.8f})\n")
#     time.sleep(1)

# # Print the best found parameter
# best_index = np.argmin([e[1] for e in evaluations])
# best_param, best_cost = evaluations[best_index]
# print(f"\nBest parameter found: {best_param:.8f} with cost {best_cost:.8f}")
