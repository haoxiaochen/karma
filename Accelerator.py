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
        self.readonly = False
        # size
        self.size = self.data["size"]
        self.tile_X = cfg["Arch"]["NumPEs"][0]
        self.tile_Y = cfg["Arch"]["NumPEs"][1]
        self.Z_depth = self.cfg["Mem"]["Depth"]
        self.dim0_extent = self.data['diag_A'].shape[0]
        self.iters = math.ceil(self.dim0_extent / self.Z_depth)
        # double buffering with the size of 2*depth
        self.domain_vec_in = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        self.domain_vec_out = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        self.domain_diag_mtx = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        self.domain_index = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        # index
        self.put_dim0_idx = 0
        self.put_dim0_idx_b = 0
        self.read_dim0_idx = 0

    def get_read_size(self):
        depth = min((self.dim0_extent - self.put_dim0_idx), self.Z_depth)
        access_size = (self.tile_X * self.tile_Y) * depth
        return 2 * access_size

    def get_write_size(self):
        depth = min((self.dim0_extent - self.read_dim0_idx), self.Z_depth)
        access_size = (self.tile_X * self.tile_Y) * depth
        return access_size

    def get_previous(self):
        # Get x from the current execution
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.read_dim0_idx))):
            for i in range(self.tile_X):
                for j in range(self.tile_Y):
                    value, ijk_index = yield self.domain_vec_out[i][j].get()
                    if ijk_index[0] < self.size[0] and ijk_index[1] < self.size[1] and \
                        ijk_index[2] < self.size[2]:
                        self.data["x"][ijk_index] = value
                        logger.trace(f"(Cycle {self.env.now}) DomainData: get x{ijk_index}={value} from PE({i}, {j})")
                    else:
                        logger.trace(f"(Cycle {self.env.now}) DomainData: get out-of-bound x{ijk_index}={value} from PE({i}, {j}), ignored")
            self.read_dim0_idx += 1

    def put_next(self):
        proc_put_b = self.env.process(self.put_next_b())
        proc_put_diagA = self.env.process(self.put_next_diagA())
        yield proc_put_b & proc_put_diagA

    def put_next_b(self):
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.put_dim0_idx_b))):
            for i in range(self.tile_X):
                for j in range(self.tile_Y):
                    # wait until b is valid (on halo and wholly updated by HEU)
                    while self.data["b_valid"][self.put_dim0_idx_b][i][j] > 0:
                        # ijk = self.data["ijk"][self.put_dim0_idx_b][i][j]
                        # print(f"waiting for b{ijk} to be valid, current valid state={self.data["b_valid"][self.put_dim0_idx_b][i][j]}")
                        yield self.env.timeout(1)
                    yield self.domain_vec_in[i][j].put(self.data["b"][self.put_dim0_idx_b][i][j])
                    logger.trace(f"(Cycle {self.env.now}) DomainData: put b of row {self.data['ijk'][self.put_dim0_idx_b][i][j]} to PE({i}, {j})")
            self.put_dim0_idx_b += 1

    def put_next_diagA(self):
        # Put data for the next execution
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.put_dim0_idx))):
            for i in range(self.tile_X):
                for j in range(self.tile_Y):
                    yield self.domain_diag_mtx[i][j].put(self.data["diag_A"][self.put_dim0_idx][i][j])
                    yield self.domain_index[i][j].put(self.data["ijk"][self.put_dim0_idx][i][j])
                    logger.trace(f"(Cycle {self.env.now}) DomainData: put Ajj & ijk of row {self.data['ijk'][self.put_dim0_idx][i][j]} to PE({i}, {j})")
            self.put_dim0_idx += 1


class DomainSpMatData:
    def __init__(self, env, cfg, data):
        self.env = env
        self.cfg = cfg
        self.data = data
        self.readonly = True
        # size
        self.size = self.data["size"]
        self.tile_X = cfg["Arch"]["NumPEs"][0]
        self.tile_Y = cfg["Arch"]["NumPEs"][1]
        self.Z_depth = self.cfg["Mem"]["Depth"]
        self.dim0_extent = self.data['A'].shape[0]
        self.iters = math.ceil(self.dim0_extent / self.Z_depth)
        # double buffering with the size of 2*depth
        self.domain_mtx = [[simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.tile_Y)] for _ in range(self.tile_X)]
        # index
        self.put_dim0_idx = 0

    def get_read_size(self):
        depth = min((self.dim0_extent - self.put_dim0_idx), self.Z_depth)
        access_size = (self.tile_X * self.tile_Y) * depth
        num_points = self.data["A"].shape[-1]
        return num_points * access_size

    def put_next(self):
        # Put data for the next execution
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.put_dim0_idx))):
            for i in range(self.tile_X):
                for j in range(self.tile_Y):
                    yield self.domain_mtx[i][j].put(self.data["A"][self.put_dim0_idx][i][j])
                    logger.trace(f"(Cycle {self.env.now}) DomainSpMatData: put spmat data of row to PE({i}, {j}) with index={self.put_dim0_idx}")
            self.put_dim0_idx += 1

class HaloData:
    def __init__(self, env, cfg, data, position): # X:0, Y:1
        self.env = env
        self.cfg = cfg
        self.data = data
        self.position = position
        self.readonly = False
        # size
        self.size = self.data["size"]
        self.tile_X = cfg["Arch"]["NumPEs"][0]
        self.tile_Y = cfg["Arch"]["NumPEs"][1]
        self.Z_depth = self.cfg["Mem"]["Depth"]

        self.dim0_extent = math.ceil(self.size[0] / self.tile_X) * math.ceil(self.size[1] / self.tile_Y) * int(self.size[2])
        self.iters = math.ceil(self.dim0_extent / self.Z_depth)
        self.halo_points = get_num_halo_points(cfg["StencilType"], cfg["NumDims"], position, self.tile_Y if position==0 else self.tile_X)
        # double buffering with the size of 2*depth
        self.halo_vec_in = [simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.halo_points)]
        self.halo_idx_in = [simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.halo_points)]
        self.halo_vec_out = [simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.halo_points)]
        self.halo_idx_out = [simpy.Store(env, capacity=2*self.Z_depth) for _ in range(self.halo_points)]
        # index
        self.put_dim0_idx = 0
        self.read_dim0_idx = 0
        self.stencil_type = cfg["StencilType"]

    def get_read_size(self):
        depth = min((self.dim0_extent - self.put_dim0_idx), self.Z_depth)
        return self.halo_points * depth

    def get_write_size(self):
        depth = min((self.dim0_extent - self.read_dim0_idx), self.Z_depth)
        return self.halo_points * depth

    def get_previous(self):
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.read_dim0_idx))):
            for i in range(self.halo_points):
                value, index, out_flag, agg_flag = yield self.halo_vec_out[i].get()
                ijk_index = yield self.halo_idx_out[i].get()
                if index[0] >= 0:
                    self.data["b"][index] = value
                    if self.position == 0: # out_i | agg_i
                        self.data["b_valid"][index] -= ((out_flag << 1) + (agg_flag << 2))
                    else: # out_j | agg_j
                        self.data["b_valid"][index] -= (out_flag + (agg_flag << 3))
                    logger.trace(f"(Cycle {self.env.now}) HaloData: get b{ijk_index}={value} from HEU ({self.position}, {i}) with valid state={bin(self.data["b_valid"][index])}")
            self.read_dim0_idx += 1

    def put_next(self):
        for _ in range(int(min(self.Z_depth, self.dim0_extent - self.put_dim0_idx))):
            for i in range(self.halo_points):
                halo = "halo_x" if self.position == 0 else "halo_y"
                index = self.data[halo][self.put_dim0_idx][i]

                # b_valid共4bit，bit[0]表示需要通过out_j更新，bit[1]表示需要通过out_i更新
                # bit[2]表示需要通过agg_i更新，bit[3]表示需要通过agg_j更新
                # 优先级从高bit到低bit，这是按列优先的调度顺序决定的
                # agg_j agg_i out_i out_j
                if index[0] >= 0:
                    if self.stencil_type == 0: # out_i -> out_j
                        if self.position == 1: # out_j
                            while self.data["b_valid"][index] & 0b0010 != 0: # wait out_i
                                yield self.env.timeout(1)

                    elif self.stencil_type == 1: # (agg_i | out_i) -> (agg_j | out_j)
                        if self.position == 1: # (agg_j | out_j)
                            while self.data["b_valid"][index] & 0b0110 != 0: # wait agg_i & out_i
                                yield self.env.timeout(1)

                    elif self.stencil_type == 2: # agg_i -> out_i -> out_j
                        if self.position == 0 and i == 0: # out_i
                            while self.data["b_valid"][index] & 0b0100 != 0: # wait agg_i
                                yield self.env.timeout(1)
                        elif self.position == 1: # out_j
                            while self.data["b_valid"][index] & 0b0110 != 0: # wait agg_i & out_i
                                yield self.env.timeout(1)

                    elif self.stencil_type == 3: # agg_j -> agg_i -> out_i -> out_j
                        ijk = self.data["ijk"][index]
                        if self.position == 0 and i == 0: # out_i
                            while self.data["b_valid"][index] & 0b1100 != 0: # wait agg_i & agg_j
                                yield self.env.timeout(1)
                                # print(f"b{ijk} valid_state={bin(self.data["b_valid"][index])} is Waiting for agg_i & agg_j")
                        elif self.position == 0 and i != 0: # agg_i
                            while self.data["b_valid"][index] & 0b1000 != 0: # wait agg_j
                                yield self.env.timeout(1)
                                # print(f"b{ijk} valid_state={bin(self.data["b_valid"][index])} Waiting for agg_j")
                        elif self.position == 1 and i % 2 == 0: # out_j
                            while self.data["b_valid"][index] & 0b1110 != 0: # wait agg_ij & out_i
                                yield self.env.timeout(1)
                                # print(f"b{ijk} valid_state={bin(self.data["b_valid"][index])} Waiting for agg_i & agg_j & out_i")

                value = self.data["b"][index]
                ijk = self.data["ijk"][index]
                yield self.halo_vec_in[i].put((value, index))
                yield self.halo_idx_in[i].put(ijk)
                if index[0] >= 0:
                    logger.trace(f"(Cycle {self.env.now}) HaloData: put b{ijk}={value} to HEU ({self.position}, {i}) with valid state={bin(self.data["b_valid"][index])}")
                else:
                    logger.trace(f"(Cycle {self.env.now}) HaloData: put invalid b to HEU ({self.position}, {i}) with valid state={bin(self.data["b_valid"][index])}")

            self.put_dim0_idx += 1


class Accelerator:
    def __init__(self, env, cfg, data, progressbar=False):
        self.env = env
        self.cfg = cfg
        self.data = data
        self.progressbar = progressbar
        # Memory

        # 由于数据深度不同所以为A单独设置DRAM
        self.domain_spmat_data = DomainSpMatData(env, cfg, data)
        self.domain_spmat_dram = DRAM(env, "domain_spmat_data", cfg["Mem"]["SpMat_DRAM_BW"], self.domain_spmat_data)
        self.domain_data = DomainData(env, cfg, data)
        self.domain_dram = DRAM(env, "domain_data", cfg["Mem"]["Domain_DRAM_BW"], self.domain_data)
        self.halo_data_X = HaloData(env, cfg, data, 0)
        self.halo_dram_X = DRAM(env, "halo_x", cfg["Mem"]["Halo_DRAM_BW"], self.halo_data_X)
        self.halo_data_Y = HaloData(env, cfg, data, 1)
        self.halo_dram_Y = DRAM(env, "halo_y", cfg["Mem"]["Halo_DRAM_BW"], self.halo_data_Y)
        self.buffers = Buffers(env, cfg)
        # PE array
        self.tile_X = cfg["Arch"]["NumPEs"][0]
        self.tile_Y = cfg["Arch"]["NumPEs"][1]
        self.boundary_x = [BoundaryPorts(env) for _ in range(self.tile_Y)]  # X:1, Y:0
        self.boundary_y = [BoundaryPorts(env) for _ in range(self.tile_X)]
        self.PE_Array = PEArray(env, cfg, self.buffers, self.domain_data, self.domain_spmat_data, [self.boundary_x, self.boundary_y])
        # HEU
        self.HEU_X = HEU(env, cfg, self.buffers, self.halo_data_X, self.boundary_x, 0)  # X:0, Y:1
        self.HEU_Y = HEU(env, cfg, self.buffers, self.halo_data_Y, self.boundary_y, 1)
        self.actions = [env.process(self.run())]

    def wait_for_finish(self):
        yield self.domain_dram.proc_write

    def run(self):
        bar = progressbar.ProgressBar(maxval=self.domain_data.iters * self.domain_data.Z_depth)
        if self.progressbar: bar.start()
        while True:
            yield self.env.timeout(1)  # Simulate processing time
            if self.progressbar: bar.update(self.domain_data.read_dim0_idx)

    def check_correctness(self):
        for i in range(self.data["size"][0]):
            for j in range(self.data["size"][1]):
                for k in range(self.data["size"][2]):
                    if self.data["x"][i][j][k] != 1.0:
                        # print(f"{i}, {j}, {k}, {self.data["x"][i][j][k]}")
                        return False
        return True

    def print(self):
        wall_clock_time = self.env.now / (self.cfg["Freq"]*1e9)
        # add mul div
        stencil_type = self.cfg["StencilType"]
        stencil_points = self.data['A'].shape[-1]
        vec_lanes = self.cfg['Arch']['VecLanes']
        mul_cycles = math.ceil(stencil_points / vec_lanes)
        delay_arr = [
            [1, mul_cycles, 1],
            [2, mul_cycles, 1],
            [1, mul_cycles, 1],
            [1, mul_cycles, 1],
        ]

        delay = self.cfg['Delay']['Add'] * delay_arr[stencil_type][0] + \
                self.cfg['Delay']['Mul'] * delay_arr[stencil_type][1] + \
                self.cfg['Delay']['Div'] * delay_arr[stencil_type][2]

        ideal_cycles = math.ceil(math.prod(self.data["size"]) / (self.tile_X * self.tile_Y)) * delay

        domain_counter = self.domain_dram.counter + self.domain_spmat_dram.counter
        halo_counter = self.halo_dram_X.counter + self.halo_dram_Y.counter
        dram_energy = (domain_counter + halo_counter) * self.cfg["Energy"]["DRAM"]
        domain_vector = self.buffers.domain_vec_in.counter + self.buffers.domain_vec_out.counter
        halo_vector = self.buffers.halo_vec_in.counter + self.buffers.halo_vec_out.counter
        sram_energy = (domain_vector + halo_vector + self.buffers.domain_mtx.counter + self.buffers.domain_diag_mtx.counter)*self.cfg["Energy"]["SRAM"]

        PE_mul, PE_div, PE_add = self.PE_Array.stat()
        HEU_add = self.HEU_X.add_counter + self.HEU_Y.add_counter
        add_energy = (PE_add + HEU_add)*self.cfg["Energy"]["Add"]
        mul_energy = PE_mul*self.cfg["Energy"]["Mul"]
        div_energy = PE_div*self.cfg["Energy"]["Div"]
        compute_energy = add_energy + mul_energy + div_energy
        total_energy = dram_energy + sram_energy + compute_energy

        print("="*80)
        print(" "*30, "SIMULATION REPORT", " "*30)
        print("="*80)
        print(f"Total Cycles: {self.env.now}")
        print(f"Wallclock Time: {wall_clock_time * 1000:.2f} ms")
        print(f"Ideal delay: {delay}, Ideal Cycles: {ideal_cycles}")
        print(f"PE Utilization: {ideal_cycles / self.env.now * 100:.2f}%")
        print('-'*35, "DRAM", '-'*35)
        print(f"Domain Access Volume: {domain_counter},\t\t\tAverage BW: {(domain_counter*8/1e9)/wall_clock_time:.2f} GB/s")
        print(f"Halo Access Volume: {halo_counter},\t\t\tAverage BW: {(halo_counter*8/1e9)/wall_clock_time:.2f} GB/s")
        print(f"Energy: {dram_energy} pJ ({dram_energy / total_energy * 100:.2f}%)")
        print('-'*35, "SRAM", '-'*35)
        print(f"Domain Vector Access Volume: {domain_vector},\t\tAverage BW: {(domain_vector*8/1e9)/wall_clock_time:.2f} GB/s")
        print(f"Halo Vector Access Volume: {halo_vector},\t\tAverage BW: {(halo_vector*8/1e9)/wall_clock_time:.2f} GB/s")
        print(f"Domain Matrix Access Volume: {self.buffers.domain_mtx.counter},\t\tAverage BW: {(self.buffers.domain_mtx.counter*8/1e9)/wall_clock_time:.2f} GB/s")
        print(f"Domain Diagonal Access Volume: {self.buffers.domain_diag_mtx.counter},\t\tAverage BW: {(self.buffers.domain_diag_mtx.counter*8/1e9)/wall_clock_time:.2f} GB/s")
        print(f"Energy: {sram_energy} pJ ({sram_energy / total_energy * 100:.2f}%)")
        print('-'*34, "PE/HEU", '-'*34)
        print(f"PE Add: {PE_add}, Mul: {PE_mul}, Div: {PE_div}")
        print(f"HEU Add: {HEU_add}")
        print(f"Energy: {compute_energy} pJ ({compute_energy / total_energy * 100:.2f}%)")
        print("="*80 + "\n")