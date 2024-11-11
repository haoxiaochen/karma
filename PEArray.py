import simpy
import math
from loguru import logger
from MtxGen import get_num_domain_points

class PEPorts:
    def __init__(self, env):
        # output ports
        self.out_i = simpy.Store(env, capacity=1)
        self.out_j = simpy.Store(env, capacity=1)
        self.agg_out_i = simpy.Store(env, capacity=1)
        self.agg_out_j = simpy.Store(env, capacity=1)
        # input ports
        self.in_i = simpy.Store(env, capacity=1)
        self.in_j = simpy.Store(env, capacity=1)
        self.agg_in_i = simpy.Store(env, capacity=1)
        self.agg_in_j = simpy.Store(env, capacity=1)

class BoundaryPorts:
    def __init__(self, env):
        self.out = simpy.Store(env, capacity=1)
        self.agg_out = simpy.Store(env, capacity=1)

class PE:
    def __init__(self, env, cfg, bufs, data, ports, i, j):
        self.env = env
        self.cfg = cfg
        self.bufs = bufs
        self.data = data
        self.ports = ports
        self.stencil_type = cfg["StencilType"]
        self.dims = cfg["NumDims"]
        # PE index
        self.i = i
        self.j = j
        # computes
        self.mul_counter = 0
        self.div_counter = 0
        self.add_counter = 0
        # internal control
        self.new_x = simpy.Store(env, capacity=1)
        self.R_k = simpy.Store(env, capacity=1)
        self.vec_results = simpy.Store(env, capacity=1)
        self.actions = [env.process(self.ScalarUnit()), env.process(self.VectorUnit()), env.process(self.Aggregator())]
        # internal data
        self.shift_x = []

    def ScalarUnit(self):
        while True:
            # Get data from the buffer
            tick = self.env.now
            yield self.env.process(self.bufs.domain_vec_in.access(2))
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get domain vec access ready takes {self.env.now - tick} cycles")

            b = yield self.data.domain_vec_in[self.i][self.j].get()
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get domain vec data ready takes {self.env.now - tick} cycles")
            aii = yield self.data.domain_diag_mtx[self.i][self.j].get()
            index = yield self.data.domain_index[self.i][self.j].get()
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get diag and index data ready takes {self.env.now - tick} cycles")

            in_i = (yield self.ports.in_i.get()) if self.i != 0 else 0
            in_j = (yield self.ports.in_j.get()) if self.j != 0 else 0
            R_k = (yield self.R_k.get()) if index[2] != 0 else 0
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get data ready takes {self.env.now - tick} cycles")

            sum = in_i + in_j + R_k
            x = (b - sum) / aii
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: compute variable x{index}={x} with aii={aii}, b={b}, in_i={in_i}, in_j={in_j}, R_k={R_k}")
            self.add_counter += 4
            self.div_counter += 1
            delay = 2 * self.cfg["Delay"]["Add"] + self.cfg["Delay"]["Div"]
            yield self.env.timeout(delay)
            yield self.new_x.put((x, index))

            yield self.env.process(self.bufs.domain_vec_out.access(1))
            yield self.data.domain_vec_out[self.i][self.j].put((x, index))
            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: one iteration takes {self.env.now - tick} cycles")

    def VectorUnit(self):
        while True:
            tick = self.env.now
            NumPoints = get_num_domain_points(self.stencil_type, self.dims)
            new_x = yield self.new_x.get()
            yield self.env.process(self.bufs.domain_mtx.access(NumPoints))
            vec_A = yield self.data.domain_mtx[self.i][self.j].get()

            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: get data ready takes {self.env.now - tick} cycles")

            self.shift_x.insert(0, new_x)
            if self.stencil_type == 0 and self.dims == 3: # Star7P
                if len(self.shift_x) > 1:
                    self.shift_x.pop()
                vec_x = [self.shift_x[0][0]] * 3
            elif self.stencil_type == 2 and self.dims == 3: # Diamond13P
                if len(self.shift_x) > 3:
                    self.shift_x.pop()
                vec_x = [self.shift_x[0][0]] * 3 + [self.shift_x[1][0]] * 2 + [self.shift_x[2][0]]
            vec_results = [a * x for a, x in zip(vec_A, vec_x)]
            self.mul_counter += NumPoints
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: terms={vec_results} with shift_x={self.shift_x}, vec_A={vec_A}")

            # Wait R_k, and assume R_k is the first term
            yield self.env.timeout(self.cfg["Delay"]["Mul"])
            if new_x[1][2] != self.data.size[2]-1: # Not the last point
                yield self.R_k.put(vec_results[0])
            # The remaining data are processed in pipelining
            Lanes = self.cfg["Arch"]["VecLanes"]
            delay = (NumPoints - 1) // Lanes
            if delay > 0:
                yield self.env.timeout(delay)
            yield self.vec_results.put(vec_results)
            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: one iteration takes {self.env.now - tick} cycles")

    def Aggregator(self):
        while True:
            tick = self.env.now
            vec_results = yield self.vec_results.get()
            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: get data ready takes {self.env.now - tick} cycles")

            if self.stencil_type == 0 and self.dims == 3: # Star7P
                assert len(vec_results) == 3
                yield self.ports.out_j.put(vec_results[1])
                yield self.ports.out_i.put(vec_results[2])
                logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: pass term([1])={vec_results[1]} through out_j")
                logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: pass term([2])={vec_results[2]} through out_i")
            elif self.stencil_type == 2 and self.dims == 3: # Diamond13P
                assert len(vec_results) == 6
                agg_in_i = (yield self.ports.agg_in_i) if self.i != 0 else 0
                out_j = agg_in_i + self.vec_results[2] + self.vec_results[3]
                agg_out_j = self.vec_results[4] + self.vec_results[5]
                add_times = 2 if self.cfg["Arch"]["AggAdders"] < 2 else 1
                yield self.env.timeout(add_times * self.cfg["Delay"]["Add"])
                self.add_counter += 2

                yield self.ports.out_i.put(vec_results[1])
                yield self.ports.out_j.put(out_j)
                yield self.ports.agg_out_j.put(agg_out_j)
                logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: pass term([1])={vec_results[1]} through out_i")
                logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: pass term([2]+[3]+agg_in_i)={out_j} through out_j")
                logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: pass term([4]+[5])={agg_out_j} through agg_out_j")
            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: one iteration takes {self.env.now - tick} cycles")

class PEArray:
    def __init__(self, env, cfg, bufs, data, boundaries):
        self.env = env
        self.cfg = cfg
        self.data = data
        self.boundaries = boundaries

        self.num_PEs = cfg["Arch"]["NumPEs"]
        self.ports = [[PEPorts(env) for _ in range(self.num_PEs[1])] for _ in range(self.num_PEs[0])]
        self.PEs = [[PE(env, cfg, bufs, data, self.ports[i][j], i, j) for j in range(self.num_PEs[1])] for i in range(self.num_PEs[0])]
        self.actions = [env.process(self.run(i, j)) for i in range(self.num_PEs[0]) for j in range(self.num_PEs[1])]

    def run(self, i, j):
        while True:
            stencil_type = self.cfg["StencilType"]
            use_agg_i = True if stencil_type == 2 else False    # Diamond
            use_agg_j = True if stencil_type == 3 else False    # Box

            out_i = yield self.ports[i][j].out_i.get()
            out_j = yield self.ports[i][j].out_j.get()
            if use_agg_i:
                agg_out_i = yield self.ports[i][j].agg_out_i.get()
            if use_agg_j:
                agg_out_j = yield self.ports[i][j].agg_out_j.get()
            # yield self.env.timeout(1)  # Forward delay

            if i != self.num_PEs[0]-1:
                yield self.ports[i+1][j].in_i.put(out_i)
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i+1}, {j}) through (out_i, in_i)")
                if use_agg_i:
                    yield self.ports[i+1][j].agg_in_i.put(agg_out_i)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i+1}, {j}) through (agg_out_i, agg_in_i)")
            else:
                yield self.boundaries[0][j].out.put(out_i)
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (0, {j}) through out_i")
                if use_agg_i:
                    yield self.boundaries[0][j+1].agg_out.put(agg_out_i)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (0, {j}) through agg_out_i")

            if j != self.num_PEs[1]-1:
                yield self.ports[i][j+1].in_j.put(out_j)
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i}, {j+1}) through (out_j, in_j)")
                if use_agg_j:
                    yield self.ports[i][j+1].agg_in_j.put(agg_out_j)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i}, {j+1}) through (agg_out_j, agg_in_j)")
            else:
                yield self.boundaries[1][i].out.put(out_j)
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (1, {i}) through out_j")
                if use_agg_j:
                    yield self.boundaries[1][i+1].agg_out.put(agg_out_j)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (1, {i}) through agg_out_j")
