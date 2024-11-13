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

        # debug ports
        self.out_i_ijk = simpy.Store(env, capacity=1)
        self.out_j_ijk = simpy.Store(env, capacity=1)
        self.in_i_ijk = simpy.Store(env, capacity=1)
        self.in_j_ijk = simpy.Store(env, capacity=1)

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
        self.num_points = get_num_domain_points(self.stencil_type, self.dims)
        self.vec_results = [simpy.Store(env, capacity=1) for _ in range(self.num_points)]
        self.ijk_index = simpy.Store(env, capacity=1)

        self.actions = [env.process(self.ScalarUnit()), env.process(self.VectorUnit()), env.process(self.Aggregator())]
        # internal data
        self.shift_x = [0 for _ in range(self.num_points)]
        self.agg_ijk_index = (-1, -1, -1)

        # lanes in each stage
        stage_lanes_3d = [[3], [3, 3], [3, 2, 1], [3, 2, 3, 1, 2, 1, 1]]
        stage_lanes_2d = [[2], [2, 2], [2, 1], [2, 1, 1]]
        self.stage_lanes = stage_lanes_3d if self.dims == 3 else stage_lanes_2d

        self.id2stage = []
        for i, lane_n in enumerate(self.stage_lanes[self.stencil_type]):
            self.id2stage += [i] * lane_n
        assert(len(self.id2stage) == self.num_points)
        self.term_id_dict = {f"term[{i}]": i for i in range(self.num_points)}

        self.z_offset = [[1, 0, 0],
                         [1, 0, 0, 2, 0, 0],
                         [1, 0, 0, 1, 0, 1],
                         [1, 0, 0, 1, 1, 2, 1, 2, 2, 3, 2, 3, 4]]

    def ScalarUnit(self):
        while True:
            # Get data from the buffer
            tick = self.env.now
            yield self.env.process(self.bufs.domain_vec_in.access(1))
            yield self.env.process(self.bufs.domain_diag_mtx.access(1))

            aii = yield self.data.domain_diag_mtx[self.i][self.j].get()
            ijk_index = yield self.data.domain_index[self.i][self.j].get()
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get aii, ijk_index ready takes {self.env.now - tick} cycles ijk_index={ijk_index}")

            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: waiting for b")
            b = yield self.data.domain_vec_in[self.i][self.j].get()
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get b ready takes {self.env.now - tick} cycles")

            in_j_ijk = (yield self.ports.in_j_ijk.get()) if self.j != 0 else 0
            in_j = (yield self.ports.in_j.get()) if self.j != 0 else 0
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get in_j ready takes {self.env.now - tick} cycles")

            in_i_ijk = (yield self.ports.in_i_ijk.get()) if self.i != 0 else 0
            in_i = (yield self.ports.in_i.get()) if self.i != 0 else 0
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get in_i ready takes {self.env.now - tick} cycles")

            R_k = (yield self.R_k.get()) if ijk_index[2] != 0 else 0
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get all data ready takes {self.env.now - tick} cycles")

            sum = in_i + in_j + R_k
            x = (b - sum) / aii
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: compute variable x{ijk_index}={x} with aii={aii}, b={b}, in_i={in_i}, in_j={in_j}, R_k={R_k}, in_i_ijk={in_i_ijk}, in_j_ijk={in_j_ijk}")
            self.add_counter += 3
            self.div_counter += 1
            delay = self.cfg["Delay"]["Add"] + self.cfg["Delay"]["Div"]
            yield self.env.timeout(delay)
            yield self.new_x.put((x, ijk_index))

            yield self.env.process(self.bufs.domain_vec_out.access(1))
            yield self.data.domain_vec_out[self.i][self.j].put((x, ijk_index))
            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: one iteration takes {self.env.now - tick} cycles")

    def VectorUnit(self):
        # schedule multiply index
        # 按照critical path长度降序排列
        sche_seq = [[0, 1, 2],
                    [1, 4, 2, 5, 0, 3],
                    [4, 5, 2, 3, 1, 0],
                    [10, 11, 12, 6, 8, 9, 1, 3, 5, 7, 4, 2, 0]]

        while True:
            tick = self.env.now
            new_x, ijk_index = yield self.new_x.get()
            yield self.env.process(self.bufs.domain_mtx.access(self.num_points))
            vec_A = yield self.data.domain_mtx[self.i][self.j].get()
            assert(len(vec_A) == self.num_points)

            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: get data ready takes {self.env.now - tick} cycles")

            # shift x
            self.shift_x.insert(0, new_x)
            self.shift_x.pop()

            vec_x = []
            for shift_x, lane_n in zip(self.shift_x, self.stage_lanes[self.stencil_type]):
                vec_x += [shift_x] * lane_n

            yield self.ijk_index.put(ijk_index)
            # schedule
            lanes = self.cfg["Arch"]["VecLanes"]
            sche_cycles = (self.num_points + lanes - 1) // lanes
            for i in range(sche_cycles):
                # pipeline
                if i == 0:
                    yield self.env.timeout(self.cfg["Delay"]["Mul"])
                else:
                    yield self.env.timeout(1)

                sche_ids = []
                for j in range(lanes):
                    id = i * lanes + j
                    if id >= self.num_points:
                        break
                    sche_id = sche_seq[self.stencil_type][id]
                    yield self.vec_results[sche_id].put(vec_x[sche_id] * vec_A[sche_id])
                    sche_ids.append(sche_id)

                logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: veclen={len(sche_ids)} terms={sche_ids} with shift_x={self.shift_x}, vec_A={vec_A}")

            self.mul_counter += self.num_points
            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: one iteration takes {self.env.now - tick} cycles")

    def Aggregator(self):
        # bind name to port for debugging
        agg_in_i = (self.ports.agg_in_i, "agg_in_i")
        agg_in_j = (self.ports.agg_in_j, "agg_in_j")
        agg_out_i = (self.ports.agg_out_i, "agg_out_i")
        agg_out_j = (self.ports.agg_out_j, "agg_out_j")
        out_i = (self.ports.out_i, "out_i")
        out_j = (self.ports.out_j, "out_j")
        rk = (self.R_k, "R_k")
        vec_results = [(self.vec_results[i], f"term[{i}]") for i in range(self.num_points)]

        # generate adder & buf processes
        while True:
            tick = self.env.now
            # logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: get data ready takes {self.env.now - tick} cycles")
            self.agg_ijk_index = yield self.ijk_index.get()
            self.dim0_index = yield self.data.domain_dim0_idx[self.i][self.j].get()

            if self.dims == 3:
                # A_vec_target [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
                if self.stencil_type == 0: # Star7P
                    out_i_proc = self.env.process(self.buf(vec_results[2], out_i))
                    out_j_proc = self.env.process(self.buf(vec_results[1], out_j))
                    rk_proc = self.env.process(self.buf(vec_results[0], rk))
                    yield out_i_proc & out_j_proc & rk_proc

                # A_vec_target [(0, 0, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, -1), (0, 2, -1)]
                elif self.stencil_type == 1: # Star13P
                    out_i_proc = self.env.process(self.adder([agg_in_i, vec_results[1]], out_i))
                    out_j_proc = self.env.process(self.adder([agg_in_j, vec_results[2]], out_j))
                    rk_proc = self.env.process(self.adder([vec_results[0], vec_results[3]], rk))
                    agg_out_i_proc = self.env.process(self.buf(vec_results[4], agg_out_i))
                    agg_out_j_proc = self.env.process(self.buf(vec_results[5], agg_out_j))
                    yield out_i_proc & out_j_proc & rk_proc & agg_out_i_proc & agg_out_j_proc

                # A_vec_target [(0, 0, 1), (1, 0, 0), (0, 1, 0),
                #               (0, 1, 0), (1, 1, -1),
                #               (1, 1, -1)]
                elif self.stencil_type == 2: # diamond13P
                    out_i_proc = self.env.process(self.buf(vec_results[1], out_i))
                    out_j_proc = self.env.process(self.adder([agg_in_i, vec_results[2], vec_results[3]], out_j))
                    rk_proc = self.env.process(self.buf(vec_results[0], rk))
                    agg_out_i_proc = self.env.process(self.adder([vec_results[4], vec_results[5]], agg_out_i))
                    yield out_i_proc & out_j_proc & rk_proc & agg_out_j_proc

                # A_vec_target [(0, 0, 1), (1, 0, 0), (0, 1, 0),
                #               (0, 1, 0), (1, 0, 0),
                #               (0, 1, 0), (1, 1, -1), (1, 0, 0),
                #               (1, 1, -1),
                #               (1, 1, -1), (1, 2, -2),
                #               (1, 2, -2),
                #               (1, 2, -2)]
                elif self.stencil_type == 3: # box27P
                    out_i_proc = self.env.process(self.adder([vec_results[1], vec_results[4], vec_results[7]], out_i))
                    out_j_proc = self.env.process(self.adder([agg_in_i, vec_results[2], vec_results[3], vec_results[5]], out_j))
                    agg_out_i_proc = self.env.process(self.adder([agg_in_j, vec_results[6], vec_results[8], vec_results[9]], agg_out_i))
                    agg_out_j_proc = self.env.process(self.adder([vec_results[10], vec_results[11], vec_results[12]], agg_out_j))
                    rk_proc = self.env.process(self.buf(vec_results[0], rk))
                    yield out_i_proc & out_j_proc & rk_proc & agg_out_i_proc & agg_out_j_proc
            else:
                pass

            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: one iteration takes {self.env.now - tick} cycles")


    # inputs [(port, name)] output (port, name)
    def adder(self, inputs, output):
        assert(self.agg_ijk_index[0] != -1)
        sum = 0
        add_n = 0
        input_str = ""
        output_port, output_name = output
        for in_port, in_name in inputs:
            if (in_name == "agg_in_i" and self.i == 0) or (in_name == "agg_in_j" and self.j == 0):
                continue
            # 处理两个tile交接的情况，避免当前tile的结果累加到下一个tile
            # (这部分没用，因为会被这部分的Amat会被自动pad 0)
            # if (in_name.startswith("term")):
            #     z = self.data.size[2]
            #     id = self.term_id_dict[in_name]
            #     cur_z_index = self.agg_ijk_index[2] - self.id2stage[id]
            #     target_z_index = cur_z_index + self.z_offset[self.stencil_type][id]


            #     if target_z_index >= z:
            #         yield in_port.get()
            #     else:
            #         input_data = yield in_port.get()
            #         sum += input_data
            #         input_str += (in_name + "+")
            #         add_n += 1
            if (in_name.startswith("term")):
                id = self.term_id_dict[in_name]
                cur_z_index = self.dim0_index - self.id2stage[id]
                target_z_index = cur_z_index + self.z_offset[self.stencil_type][id]

                if target_z_index < 0:
                    yield in_port.get()
                    continue

            input_data = yield in_port.get()
            sum += input_data
            input_str += (in_name + "+")
            add_n += 1

        if output_name != 'R_k' or self.agg_ijk_index[2] != self.data.size[2] - 1:
            if add_n >= 2:
                yield self.env.timeout(self.cfg['Delay']['Add'])
            yield output_port.put(sum)
            self.add_counter += add_n - 1
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) agg_index: {self.agg_ijk_index} Aggregator: pass {input_str}={sum} through {output_name}")
        else:
            # pass
            logger.debug(f"Aggregator: Put nothing to the outport, Output name: {output_name}, {self.agg_ijk_index[2]}")

    def buf(self, input, output):
        input_port, input_name = input
        output_port, output_name = output
        input_data = yield input_port.get()

        if output_name != 'R_k' or self.agg_ijk_index[2] != self.data.size[2] - 1:
            yield output_port.put(input_data)
            if output_name == "out_j":
                yield self.ports.out_j_ijk.put(self.agg_ijk_index)
            elif output_name == "out_i":
                yield self.ports.out_i_ijk.put(self.agg_ijk_index)
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) agg_index: {self.agg_ijk_index} Aggregator: pass {input_name}={input_data} through {output_name}")
        else:
            logger.debug(f"Aggregator: Put nothing to the outport, Output name: {output_name}, {self.agg_ijk_index[2]}")

class PEArray:
    def __init__(self, env, cfg, bufs, data, boundaries):
        self.env = env
        self.cfg = cfg
        self.data = data
        self.boundaries = boundaries

        self.num_PEs = cfg["Arch"]["NumPEs"]
        self.ports = [[PEPorts(env) for _ in range(self.num_PEs[1])] for _ in range(self.num_PEs[0])]
        self.PEs = [[PE(env, cfg, bufs, data, self.ports[i][j], i, j) for j in range(self.num_PEs[1])] for i in range(self.num_PEs[0])]

        self.actions = []
        for i in range(self.num_PEs[0]):
            for j in range(self.num_PEs[1]):
                self.actions += [env.process(self.run_i(i, j)), env.process(self.run_j(i, j))]

    def stat(self):
        mul_counter = sum([pe.mul_counter for row in self.PEs for pe in row])
        div_counter = sum([pe.div_counter for row in self.PEs for pe in row])
        add_counter = sum([pe.add_counter for row in self.PEs for pe in row])
        return mul_counter, div_counter, add_counter

    def run_i(self, i, j):
        while True:
            stencil_type = self.cfg["StencilType"]
            use_agg_i = True if stencil_type == 2 else False    # Diamond

            out_i = yield self.ports[i][j].out_i.get()
            if use_agg_i:
                agg_out_i = yield self.ports[i][j].agg_out_i.get()
            # yield self.env.timeout(1)  # Forward delay
            # debug
            out_i_ijk = yield self.ports[i][j].out_i_ijk.get()

            if i != self.num_PEs[0]-1:
                yield self.ports[i+1][j].in_i.put(out_i)
                yield self.ports[i + 1][j].in_i_ijk.put(out_i_ijk)
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i+1}, {j}) through (out_i, in_i), out_i_ijk={out_i_ijk}")
                if use_agg_i:
                    yield self.ports[i+1][j].agg_in_i.put(agg_out_i)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i+1}, {j}) through (agg_out_i, agg_in_i)")
            else:
                yield self.boundaries[0][j].out.put((out_i, out_i_ijk))
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (0, {j}) through out_i, out_i_ijk={out_i_ijk}")
                if use_agg_i:
                    yield self.boundaries[0][j].agg_out.put(agg_out_i)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (0, {j}) through agg_out_i")


    def run_j(self, i, j):
        while True:
            stencil_type = self.cfg["StencilType"]
            use_agg_j = True if stencil_type == 3 else False    # Box

            out_j = yield self.ports[i][j].out_j.get()
            if use_agg_j:
                agg_out_j = yield self.ports[i][j].agg_out_j.get()
            # yield self.env.timeout(1)  # Forward delay

            # debug
            out_j_ijk = yield self.ports[i][j].out_j_ijk.get()

            if j != self.num_PEs[1]-1:
                yield self.ports[i][j+1].in_j.put(out_j)
                yield self.ports[i][j + 1].in_j_ijk.put(out_j_ijk)
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i}, {j+1}) through (out_j, in_j), out_j_ijk={out_j_ijk}")
                if use_agg_j:
                    yield self.ports[i][j+1].agg_in_j.put(agg_out_j)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i}, {j+1}) through (agg_out_j, agg_in_j)")
            else:
                yield self.boundaries[1][i].out.put((out_j, out_j_ijk))
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (1, {i}) through out_j, out_j_ijk={out_j_ijk}")
                if use_agg_j:
                    yield self.boundaries[1][i].agg_out.put(agg_out_j)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (1, {i}) through agg_out_j")