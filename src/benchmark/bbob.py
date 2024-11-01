import cocoex as ex
import sys
from benchmark.benchmarks import benchmark_base
suite = ex.Suite("bbob", "", "")

class f01_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = -79.480000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f001_i01_d' + str(dimension).zfill(2))

class f02_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = 209.880000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f002_i01_d' + str(dimension).zfill(2))

class f03_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = 462.090000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f003_i01_d' + str(dimension).zfill(2))

class f04_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = 462.090000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f004_i01_d' + str(dimension).zfill(2))

class f05_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = 9.210000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f005_i01_d' + str(dimension).zfill(2))

class f06_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = -35.900000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f006_i01_d' + str(dimension).zfill(2))

class f07_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = -92.940000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f007_i01_d' + str(dimension).zfill(2))

class f08_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = -149.150000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f008_i01_d' + str(dimension).zfill(2))

class f09_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = -123.830000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f009_i01_d' + str(dimension).zfill(2))

class f10_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = 54.940000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f010_i01_d' + str(dimension).zfill(2))

class f11_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = -76.270000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f011_i01_d' + str(dimension).zfill(2))

class f12_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = 621.110000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f012_i01_d' + str(dimension).zfill(2))

class f13_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = -29.970000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f013_i01_d' + str(dimension).zfill(2))

class f14_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = 52.350000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f014_i01_d' + str(dimension).zfill(2))

class f15_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = -1000.000000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f015_i01_d' + str(dimension).zfill(2))

class f16_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = -71.350000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f016_i01_d' + str(dimension).zfill(2))

class f17_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = 16.940000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f017_i01_d' + str(dimension).zfill(2))

class f18_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = 16.940000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f018_i01_d' + str(dimension).zfill(2))

class f19_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = 102.550000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f019_i01_d' + str(dimension).zfill(2))

class f20_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = 546.500000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f020_i01_d' + str(dimension).zfill(2))

class f21_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = -40.780000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f021_i01_d' + str(dimension).zfill(2))

class f22_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = 1000.000000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f022_i01_d' + str(dimension).zfill(2))

class f23_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = -6.870000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f023_i01_d' + str(dimension).zfill(2))

class f24_i01(benchmark_base):
    def __init__(self, dimension=2):
        self.best = -102.610000
        super().__init__(dimension)
        self.function = suite.get_problem('bbob_f024_i01_d' + str(dimension).zfill(2))