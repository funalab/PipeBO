import numpy as np

class benchmark_base():
    def __init__(self, dimension):
        self.dimension = dimension

    def func(self, **paradict):
        para_array = np.array(list(paradict.values()))
        return self.f(para_array)

    def bbob_f(self, x):
        x = x[0]
        x_list = x.tolist()
        return -self.function(x_list)