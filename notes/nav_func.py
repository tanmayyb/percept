import numpy as np
import matplotlib.pyplot as plt

# Globals
K = 3
obstacles = [
    {'center': np.array([-8, 5]), 'radius': 5, 'distanceInfluence': 4},
    {'center': np.array([0, -9]), 'radius': 4, 'distanceInfluence': 20},
    {'center': np.array([5, 0]), 'radius': 4, 'distanceInfluence': 20}
]
goal = np.array([15, 0])
start = np.array([[-5, -15, -10, 0, -5, -10], [-5, 5, 12, 3, -15, 15]])
world = {'center': np.array([0, 0]), 'radius': 20, 'distanceInfluence': 2}
maxSteps = 60000
epsilon = 1
mag = 0.1

def plot_contour(with_path=True):
    x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    xx, yy = np.meshgrid(x, y)
    UNav = get_potential(xx, yy)

    plt.figure()
    plt.contour(xx, yy, UNav)
    plt.colormaps()

    for obstacle in obstacles:
        plt.plot(obstacle['center'][0], obstacle['center'][1], 'b.')

    plt.plot(goal[0], goal[1], "r*")

    if with_path:
        plot_path()
    plt.show()

def plot_path():
    path_colors = ['r', 'b', 'g', 'm', 'c', 'k', 'y']
    if start.shape[1] > 0:
        for i in range(start.shape[1]):
            start_ = start[:, i]
            path = calculate_path(start_)
            color = "." + "-" + path_colors[i % len(path_colors)]
            plt.plot(path[0, :], path[1, :], color, linewidth=2)
            plt.plot(start_[0], start_[1], "b*")
    else:
        path = calculate_path(start)
        plt.plot(path[0, :], path[1, :], 'b.-')
        plt.plot(start[0], start[1], "b*")
        plt.plot(path[0, :], path[1, :], '-r', linewidth=2)

def plot_surface():
    x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    xx, yy = np.meshgrid(x, y)
    UNav = get_potential(xx, yy)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, yy, UNav, cmap='viridis', edgecolor='none')
    plt.show()

def get_potential(xx, yy):
    shape = xx.shape
    UNav = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            q = np.array([xx[i, j], yy[i, j]])
            phi = calculate_phi(q)
            UNav[i, j] = phi
    return UNav

def calculate_path(start_):
    global maxSteps, epsilon, mag, goal
    if np.any(np.isnan(start_)):
        start_ = start
    starting_distance = distance(start_, goal)
    path = np.zeros((2, maxSteps))
    path[:, 0] = start_

    for step in range(1, maxSteps):
        q = path[:, step - 1]
        phi_grad = calculate_phi_grad(q)
        dist_to_goal = min(distance(q, goal), starting_distance)
        if mag is not None:
            eta = (dist_to_goal / starting_distance) ** 2
            if np.linalg.norm(phi_grad) == 0:
                break
            phi_grad = eta * mag * phi_grad / np.linalg.norm(phi_grad)
        path[:, step] = path[:, step - 1] - epsilon * phi_grad
        if dist_to_goal < 0.005:
            path = path[:, :step+1]
            break
    return path

def distance(q1, q2):
    return np.linalg.norm(q1 - q2)

def calculate_gamma(q):
    return distance(q, goal) ** (2 * K)

def calculate_gamma_grad(q):
    dist = distance(q, goal)
    if dist == 0:
        return np.zeros_like(q)
    return 2 * K * dist ** (2 * K - 1) * (q - goal) / dist

def calculate_beta(q):
    beta = world['radius'] ** 2 - distance(q, world['center']) ** 2
    for obstacle in obstacles:
        beta *= (distance(q, obstacle['center']) ** 2 - obstacle['radius'] ** 2)
    return beta

def calculate_beta_grad(q):
    num_of_obstacles = len(obstacles)
    beta_0 = world['radius'] ** 2 - distance(q, world['center']) ** 2
    beta_grad_0 = -2 * (q - world['center'])

    betas = np.ones(num_of_obstacles + 1)
    betas_grad = np.zeros((2, num_of_obstacles + 1))

    betas[0] = beta_0
    betas_grad[:, 0] = beta_grad_0

    for i, obstacle in enumerate(obstacles, start=1):
        betas[i] = distance(q, obstacle['center']) ** 2 - obstacle['radius'] ** 2
        betas_grad[:, i] = 2 * (q - obstacle['center'])

    beta_grad = np.zeros(2)
    for i in range(betas.size):
        temp = betas_grad[:, i].copy()
        for j in range(betas.size):
            if j != i:
                temp *= betas[j]
        beta_grad += temp
    return beta_grad

def calculate_alpha(q):
    gamma = calculate_gamma(q)
    beta = calculate_beta(q)
    return gamma / beta

def calculate_alpha_grad(q):
    gamma = calculate_gamma(q)
    beta = calculate_beta(q)
    gamma_grad = calculate_gamma_grad(q)
    beta_grad = calculate_beta_grad(q)
    return (gamma_grad * beta - gamma * beta_grad) / (beta ** 2)

def calculate_phi(q):
    alpha = calculate_alpha(q)
    if alpha < 0:
        return 1
    else:
        return (alpha / (1 + alpha)) ** (1 / K)

def calculate_phi_grad(q):
    alpha = calculate_alpha(q)
    alpha_grad = calculate_alpha_grad(q)
    denom = (1 + alpha)
    if denom == 0:
        return np.zeros_like(q)
    term = (alpha / denom) ** ((1 - K) / K)
    return (1 / K) * term * (1 / (denom ** 2)) * alpha_grad

# To call plotting functions:
plot_contour(True)
plot_surface()
