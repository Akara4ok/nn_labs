import numpy as np
import random

def circle_fun(x: np.ndarray, y: np.ndarray, random_bias = False, random_seed = None) -> np.ndarray:
    x_square = x * x
    y_square = y * y
    np_sum = x_square + y_square + (random.Random(random_seed) if random_bias else 0)
    return np_sum