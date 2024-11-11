import simpy
import math
import progressbar
from Memory import *
from PEArray import *
from HEU import *
from loguru import logger
from MtxGen import get_num_halo_points

class DomainData:
    def __init__(self, env, cfg, data):
        self.env = env
        self.cfg = cfg
        self.data = data
        # size
        self.size = self.data["size"]
        self.tile_X = cfg["Arch"]["NumPEs"][0]
        self.tile_Y = cfg["Arch"]["NumPEs"][1]
        self.Z_depth = self.cfg["Mem"]["Depth"]
        self.dim0_extent = math.prod(self.size) / (self.tile_X * self.tile_Y)
        self.iters = math.ceil(self.dim0_extent / self.Z_depth)
        # double buffering with the size of 2*depth
        self.domain_vec_in = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        self.domain_vec_out = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        self.domain_mtx = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        self.domain_diag_mtx = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        self.domain_index = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        # index
        self.dim0_idx = 0

    def get_read_size(self):
        access_size = (self.tile_X * self.tile_Y) * self.Z_depth
        num_points = len(self.data["A"][0])
        return (num_points + 2) * access_size

    def get_write_size(self):
        access_size = (self.tile_X * self.tile_Y) * self.Z_depth
        return access_size

    def get_previous(self):
        # Get x from the current execution
        for d in range(self.Z_depth):
            for i in range(self.tile_X):
                for j in range(self.tile_Y):
                    value, index = yield self.domain_vec_out[i][j].get()
                    self.data["x"][index] = value
                    logger.trace(f"(Cycle {self.env.now}) DomainData: get x{index}={value} from PE({i}, {j})")

    def put_next(self):
        # Put data for the next execution
        for _ in range(self.Z_depth):
            for i in range(self.tile_X):
                for j in range(self.tile_Y):
                    yield self.domain_vec_in[i][j].put(self.data["b"][self.dim0_idx][i][j])
                    yield self.domain_mtx[i][j].put(self.data["A"][self.dim0_idx][i][j])
                    yield self.domain_diag_mtx[i][j].put(self.data["diag_A"][self.dim0_idx][i][j])
                    yield self.domain_index[i][j].put(self.data["ijk"][self.dim0_idx][i][j])
                    logger.trace(f"(Cycle {self.env.now}) DomainData: put data of row {self.data['ijk'][self.dim0_idx][i][j]} to PE({i}, {j})")
            self.dim0_idx += 1

class HaloData:
    def __init__(self, env, cfg, data, position): # X:0, Y:1
        self.env = env
        self.cfg = cfg
        self.data = data
        self.position = position
        # size
        self.size = self.data["size"]
        self.tile_X = cfg["Arch"]["NumPEs"][0]
        self.tile_Y = cfg["Arch"]["NumPEs"][1]
        self.Z_depth = self.cfg["Mem"]["Depth"]
        self.dim0_extent = math.prod(self.size) / (self.tile_X * self.tile_Y)
        self.iters = math.ceil(self.dim0_extent / self.Z_depth)
        self.halo_points = get_num_halo_points(cfg["StencilType"], cfg["NumDims"], position, self.tile_Y if position==0 else self.tile_X)
        # double buffering with the size of 2*depth
        self.halo_vec_in = [simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.halo_points)]
        self.halo_idx_in = [simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.halo_points)]
        self.halo_vec_out = [simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.halo_points)]
        self.halo_idx_out = [simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.halo_points)]
        # index
        self.dim0_idx = 0

    def get_read_size(self):
        return self.halo_points * self.Z_depth

    def get_write_size(self):
        return self.halo_points * self.Z_depth

    def get_previous(self):
        # Get updated b from the current execution
        dim0_extent = math.prod(self.data["size"]) / (self.tile_X * self.tile_Y)
        for _ in range(self.Z_depth):
            for i in range(self.halo_points):
                value, index = yield self.halo_vec_out[i].get()
                ijk_index = yield self.halo_idx_out[i].get()
                if index[0] >= 0:
                    self.data["b"][index] = value
                    logger.trace(f"(Cycle {self.env.now}) HaloData: get b{ijk_index}={value} from HEU ({self.position}, {i})")

    def put_next(self):
        # Put b for the next execution
        for _ in range(self.Z_depth):
            for i in range(self.halo_points):
                halo = "halo_x" if self.position == 0 else "halo_y"
                index = self.data[halo][self.dim0_idx][i]
                value = self.data["b"][index]
                ijk = self.data["ijk"][index]
                yield self.halo_vec_in[i].put((value, index))
                yield self.halo_idx_in[i].put(ijk)
                logger.trace(f"(Cycle {self.env.now}) HaloData: put b{ijk}={value} to HEU ({self.position}, {i})")
            self.dim0_idx += 1

class Accelerator:
    def __init__(self, env, cfg, data, progressbar=False):
        self.env = env
        self.cfg = cfg
        self.data = data
        self.counter = 0
        self.progressbar = progressbar
        # Memory
        self.domain_data = DomainData(env, cfg, data)
        self.domain_dram = DRAM(env, "domain_data", cfg["Mem"]["DMBW"], self.domain_data)
        self.halo_data_X = HaloData(env, cfg, data, 0)
        self.halo_dram_X = DRAM(env, "halo_x", cfg["Mem"]["HVBW"], self.halo_data_X)
        self.halo_data_Y = HaloData(env, cfg, data, 1)
        self.halo_dram_Y = DRAM(env, "halo_y", cfg["Mem"]["HVBW"], self.halo_data_Y)
        self.buffers = Buffers(env, cfg)
        # PE array
        self.tile_X = cfg["Arch"]["NumPEs"][0]
        self.tile_Y = cfg["Arch"]["NumPEs"][1]
        self.boundary_x = [BoundaryPorts(env) for _ in range(self.tile_Y)]  # X:1, Y:0
        self.boundary_y = [BoundaryPorts(env) for _ in range(self.tile_X)]
        self.PE_Array = PEArray(env, cfg, self.buffers, self.domain_data, [self.boundary_x, self.boundary_y])
        # HEU
        self.HEU_X = HEU(env, cfg, self.buffers, self.halo_data_X, self.boundary_x, 0)  # X:0, Y:1
        self.HEU_Y = HEU(env, cfg, self.buffers, self.halo_data_Y, self.boundary_y, 1)
        self.actions = [env.process(self.run())]

    def wait_for_finish(self):
        yield self.domain_dram.proc_write

    def run(self):
        bar = progressbar.ProgressBar(maxval=self.domain_data.iters)
        if self.progressbar: bar.start()
        while True:
            logger.debug(f"(Cycle {self.env.now}) Accelerator: start a new iteration {self.counter}")
            yield self.env.timeout(1)  # Simulate processing time
            self.counter += 1
            if self.progressbar: bar.update(self.domain_data.dim0_idx)

    def check_correctness(self):
        for i in range(self.data["size"][0]):
            for j in range(self.data["size"][1]):
                for k in range(self.data["size"][2]):
                    if self.data["x"][i][j][k] != 1.0: return False
        return True

    def print(self):
        print("="*50)
        print("SIMULATION REPORT")
        print("="*50)
        print(f"Total cycles: {self.counter}")
        print(f"Domain DRAM read count: {self.domain_dram.counter_read}")
        print(f"Domain DRAM write count: {self.domain_dram.counter_write}")
        print(f"Halo DRAM X read count: {self.halo_dram_X.counter_read}")
        print(f"Halo DRAM X write count: {self.halo_dram_X.counter_write}")
        print(f"Halo DRAM Y read count: {self.halo_dram_Y.counter_read}")
        print(f"Halo DRAM Y write count: {self.halo_dram_Y.counter_write}")
        print("="*50 + "\n")