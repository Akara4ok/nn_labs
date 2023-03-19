from typing import Callable
import pandas as pd
import numpy as np
import sys

sys.path.append('Models')
from fun import circle_fun
sys.path.append('config')
from settings import RANDOM_SEED
from settings import TOTAL_COUNT


def gen(path: str, fun: Callable, n: int, a: int = 0, b: int = 10, random_bias: bool = False, random_seed: int = RANDOM_SEED) -> None:
    x = np.random.rand(n) * (b - a) + a
    y = np.random.rand(n) * (b - a) + a
    result = fun(x, y, random_bias, random_seed)
    d = {'x': x, 'y': y, 'result': result}
    df = pd.DataFrame(data=d)
    df.to_csv(path, index=False)

gen('Data/data.csv', circle_fun, TOTAL_COUNT)
