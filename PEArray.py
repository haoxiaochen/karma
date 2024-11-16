import simpy
from loguru import logger
from MtxGen import get_num_domain_points
from MtxGen import get_affine_stencil_points

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
    def __init__(self, env, cfg, bufs, data, spmat_data, ports, i, j):
        self.env = env
        self.cfg = cfg
        self.bufs = bufs
        self.data = data
        self.spmat_data = spmat_data
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
        # important, to prevent deadlock
        self.vec_fifo_depth = 4

        # internal control
        self.new_x = simpy.Store(env, capacity=1)
        self.R_k = simpy.Store(env, capacity=1)
        self.num_points = get_num_domain_points(self.stencil_type, self.dims)
        self.vec_results = [simpy.Store(env, capacity=self.vec_fifo_depth) for _ in range(self.num_points)]
        self.ijk_index = simpy.Store(env, capacity=1)

        self.actions = [env.process(self.ScalarUnit()), env.process(self.VectorUnit())]

        # lanes in each stage
        stage_lanes_3d = [[3], [3, 3], [3, 2, 1], [3, 2, 3, 1, 2, 1, 1]]
        stage_lanes_2d = [[2], [2, 2], [2, 1], [2, 1, 1]]
        self.stage_lanes = stage_lanes_3d if self.dims == 3 else stage_lanes_2d

        # internal data
        if self.dims == 3:
            self.shift_x = [(0, -(i + 1), True, (self.i, self.j, -(i + 1))) for i in range(len(self.stage_lanes[self.stencil_type]))]
        else:
            self.shift_x = [(0, -(i + 1), True, (-1, -1)) for i in range(len(self.stage_lanes[self.stencil_type]))]

        self.id2stage = []
        for i, lane_n in enumerate(self.stage_lanes[self.stencil_type]):
            self.id2stage += [i] * lane_n
        assert(len(self.id2stage) == self.num_points)
        self.term_id_dict = {f"term[{i}]": i for i in range(self.num_points)}

        self.z_offset = [
            [1, 0, 0],
            [1, 0, 0, 2, 0, 0],
            [1, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 2, 1, 2, 2, 3, 2, 3, 4]
        ]
        self.index_offset = get_affine_stencil_points(self.dims, self.stencil_type)
        self.Aggregator_init_processes()

    def ScalarUnit(self):
        while True:
            # Get data from the buffer
            tick = self.env.now
            yield self.env.process(self.bufs.domain_vec_in.access(1))
            yield self.env.process(self.bufs.domain_diag_mtx.access(1))
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get aii, ijk_index access ready takes {self.env.now - tick} cycles")

            aii = yield self.data.domain_diag_mtx[self.i][self.j].get()
            ijk_index = yield self.data.domain_index[self.i][self.j].get()
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get aii, ijk_index ready takes {self.env.now - tick} cycles ijk_index={ijk_index}")

            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: waiting for b")
            b = yield self.data.domain_vec_in[self.i][self.j].get()
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get b ready takes {self.env.now - tick} cycles")

            in_j, target_j_ijk = (yield self.ports.in_j.get()) if self.j != 0 else (0, 0)
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get in_j ready takes {self.env.now - tick} cycles")

            in_i, target_i_ijk = (yield self.ports.in_i.get()) if self.i != 0 else (0, 0)
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get in_i ready takes {self.env.now - tick} cycles")

            R_k, target_k_ijk = (yield self.R_k.get()) if ijk_index[2] != 0 else (0, 0)

            # assert(target_i_index == target_j_index and target_j_index == target_z_index)
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get all data ready takes {self.env.now - tick} cycles")

            sum = in_i + in_j + R_k
            x = (b - sum) / aii

            logger.info(f"""(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: compute variable x{ijk_index}={x} with 
                        aii={aii}, b={b}, in_i={in_i}, target_ijk={target_i_ijk}, in_j={in_j}, target_ijk={target_j_ijk}, R_k={R_k}, target_ijk={target_k_ijk}""")

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
        sche_seq_3d = [
            [0, 1, 2],
            [1, 4, 2, 5, 0, 3],
            [4, 5, 2, 3, 1, 0],
            [10, 11, 12, 6, 8, 9, 1, 3, 5, 7, 4, 2, 0]
        ]
        sche_seq_2d = [
            [0, 1],
            [2, 3, 0, 1],
            [2, 1, 0],
            [3, 2, 0, 1]
        ]
        sche_seq = sche_seq_3d if self.dims == 3 else sche_seq_2d
        dim0_index = 0
        while True:
            tick = self.env.now
            yield self.env.process(self.bufs.domain_mtx.access(self.num_points))
            vec_A, valid = yield self.spmat_data.domain_mtx[self.i][self.j].get()

            if valid and dim0_index < self.data.dim0_extent:
                logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: waiting for new_x")
                new_x, ijk_index = yield self.new_x.get()
            else:
                new_x, ijk_index = 0, (0, 0, 0)

            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: get data ready takes {self.env.now - tick} cycles")

            # shift x
            self.shift_x.insert(0, (new_x, dim0_index, valid, ijk_index))
            self.shift_x.pop()

            vec_x = []
            for shift_x, lane_n in zip(self.shift_x, self.stage_lanes[self.stencil_type]):
                vec_x += [shift_x[0]] * lane_n

            # yield self.ijk_index.put(ijk_index)
            # schedule
            lanes = self.cfg["Arch"]["VecLanes"]
            sche_cycles = (self.num_points + lanes - 1) // lanes
            mul_points = 0
            for i in range(sche_cycles):
                # pipeline
                yield self.env.timeout(self.cfg["Delay"]["Mul"] if i == 0 else 1)
                sche_ids = []
                for j in range(lanes):
                    id = i * lanes + j
                    if id >= self.num_points:
                        break
                    sche_id = sche_seq[self.stencil_type][id]
                    stage_id = self.id2stage[sche_id]

                    z = self.data.size[2]
                    z_offset = self.z_offset[self.stencil_type][sche_id] if self.dims == 3 else 0
                    cur_z = self.shift_x[stage_id][1]
                    z_target = z_offset + cur_z

                    x_valid = self.shift_x[stage_id][2]
                    cur_ijk_index = self.shift_x[stage_id][3]
                    if x_valid and z_target >= 0 and z_target < self.data.dim0_extent and \
                        (z_offset - stage_id <= 0 or z_target % z != 0):
                        # 若z_offset > stage_id说明输出到R_k, z_target % z == 0表明target与当前不在同一个tile
                        target_index = tuple(a - b for a, b in zip(cur_ijk_index, self.index_offset[sche_id]))
                        logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: waiting for port[{sche_id}] to be empty")
                        yield self.vec_results[sche_id].put((vec_x[sche_id] * vec_A[sche_id], target_index))
                        sche_ids.append(sche_id)
                        mul_points += 1
                        logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: pass terms[{sche_id}], cur_index={cur_ijk_index}, target_index={target_index}")
                    else:
                        sche_ids.append(-1)

            self.mul_counter += mul_points
            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: one iteration takes {self.env.now - tick} cycles")
            if valid:
                dim0_index += 1

    def Aggregator_init_processes(self):
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
        if self.dims == 3:
            if self.stencil_type == 0:
                self.actions += [
                    self.env.process(self.buf(vec_results[2], out_i)),
                    self.env.process(self.buf(vec_results[1], out_j)),
                    self.env.process(self.buf(vec_results[0], rk))
                ]
            elif self.stencil_type == 1:
                self.actions += [
                    self.env.process(self.adder([agg_in_i, vec_results[1]], out_i)),
                    self.env.process(self.adder([agg_in_j, vec_results[2]], out_j)),
                    self.env.process(self.adder([vec_results[0], vec_results[3]], rk)),
                    self.env.process(self.buf(vec_results[4], agg_out_i)),
                    self.env.process(self.buf(vec_results[5], agg_out_j))
                ]
            elif self.stencil_type == 2:
                self.actions += [
                    self.env.process(self.buf(vec_results[1], out_i)),
                    self.env.process(self.adder([agg_in_i, vec_results[2], vec_results[3]], out_j)),
                    self.env.process(self.buf(vec_results[0], rk)),
                    self.env.process(self.adder([vec_results[4], vec_results[5]], agg_out_i))
                ]
            elif self.stencil_type == 3:
                self.actions += [
                    self.env.process(self.adder([vec_results[1], vec_results[4], vec_results[7]], out_i)),
                    self.env.process(self.adder([agg_in_i, vec_results[2], vec_results[3], vec_results[5]], out_j)),
                    self.env.process(self.adder([agg_in_j, vec_results[6], vec_results[8], vec_results[9]], agg_out_i)),
                    self.env.process(self.adder([vec_results[10], vec_results[11], vec_results[12]], agg_out_j)),
                    self.env.process(self.buf(vec_results[0], rk))
                ]
        else:
            if self.stencil_type == 0: # star5p
                self.actions += [
                    self.env.process(self.buf(vec_results[0], out_i)),
                    self.env.process(self.buf(vec_results[1], out_j))
                ]
            # [(1, 0), (0, 1), (2, 0), (0, 2)]
            elif self.stencil_type == 1: # star7p
                self.actions += [
                    self.env.process(self.adder([agg_in_i, vec_results[0]], out_i)),
                    self.env.process(self.adder([agg_in_j, vec_results[1]], out_j)),
                    self.env.process(self.buf(vec_results[2], agg_out_i)),
                    self.env.process(self.buf(vec_results[3], agg_out_j))
                ]
            # [(1, 0), (0, 1), (1, 1)]
            elif self.stencil_type == 2: # diamond7p
                self.actions += [
                    self.env.process(self.buf(vec_results[0], out_i)),
                    self.env.process(self.adder([agg_in_i, vec_results[1]], out_j)),
                    self.env.process(self.buf(vec_results[2], agg_out_i))
                ]
            # [(1, 0), (0, 1), (1, 1), (1, 2)]
            elif self.stencil_type == 3: # box9p
                self.actions += [
                    self.env.process(self.buf(vec_results[0], out_i)),
                    self.env.process(self.adder([agg_in_i, vec_results[1]], out_j)),
                    self.env.process(self.adder([agg_in_j, vec_results[2]], agg_out_i)),
                    self.env.process(self.buf(vec_results[3], agg_out_j))
                ]


    def adder(self, inputs, output):
        output_port, output_name = output
        while True:
            sum = 0
            input_names = []
            target_indexes = []
            for in_port, in_name in inputs:
                if (in_name == "agg_in_i" and self.i == 0) or (in_name == "agg_in_j" and self.j == 0):
                    continue
                input_data, target_index = yield in_port.get()
                target_indexes.append(target_index)
                sum += input_data
                input_names.append(in_name)

            # print(target_indexes)
            # assert(len(set(target_indexes)) == 1)
            input_str = "+".join(input_names)
            yield self.env.timeout(self.cfg['Delay']['Add'])
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: waiting for output port={output_name} to be available")
            yield output_port.put((sum, target_indexes[0]))

            self.add_counter += len(inputs) - 1
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: pass {input_str}={sum} through {output_name}, target_index={target_indexes[0]}")


    def buf(self, input, output):
        input_port, input_name = input
        output_port, output_name = output
        while True:
            input_data, target_index = yield input_port.get()
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: waiting for output port={output_name} to be available")
            yield output_port.put((input_data, target_index))
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: pass {input_name}={input_data} through {output_name} target_index={target_index}")


class PEArray:
    def __init__(self, env, cfg, bufs, data, spmat_data, boundaries):
        self.env = env
        self.cfg = cfg
        self.data = data
        self.spmat_data = spmat_data
        self.boundaries = boundaries
        self.num_PEs = cfg["Arch"]["NumPEs"]
        self.ports = [[PEPorts(env) for _ in range(self.num_PEs[1])] for _ in range(self.num_PEs[0])]
        self.PEs = [[PE(env, cfg, bufs, data, spmat_data, self.ports[i][j], i, j) for j in range(self.num_PEs[1])] for i in range(self.num_PEs[0])]

        self.actions = []
        for i in range(self.num_PEs[0]):
            for j in range(self.num_PEs[1]):
                self.actions += [env.process(self.trans_out(i, j)), env.process(self.trans_aggr(i, j))]

    def stat(self):
        mul_counter = sum([pe.mul_counter for row in self.PEs for pe in row])
        div_counter = sum([pe.div_counter for row in self.PEs for pe in row])
        add_counter = sum([pe.add_counter for row in self.PEs for pe in row])
        return mul_counter, div_counter, add_counter

    def trans_out(self, i, j):
        while True:
            # trans i
            out_i, target_index = yield self.ports[i][j].out_i.get()

            if i != self.num_PEs[0]-1:
                yield self.ports[i+1][j].in_i.put((out_i, target_index))
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i+1}, {j}) through (out_i, in_i), target_ijk={target_index}")
            else:
                yield self.boundaries[0][j].out.put(out_i)
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (0, {j}) out through out_i, target_ijk={target_index}")

            # trans j
            out_j, target_index = yield self.ports[i][j].out_j.get()

            if j != self.num_PEs[1]-1:
                yield self.ports[i][j+1].in_j.put((out_j, target_index))
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i}, {j+1}) through (out_j, in_j), target_ijk={target_index}")
            else:
                yield self.boundaries[1][i].out.put(out_j)
                logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (1, {i}) out through out_j, target_ijk={target_index}")

    def trans_aggr(self, i, j):
        stencil_type = self.cfg["StencilType"]
        use_agg_i = False if stencil_type == 0 else True    # Star & Diamond & Box
        use_agg_j = True if stencil_type == 1 or stencil_type == 3 else False    # Star & Box
        if not use_agg_i and not use_agg_j:
            return
        while True:
            if use_agg_i:
                agg_out_i, target_index = yield self.ports[i][j].agg_out_i.get()
                if i != self.num_PEs[0]-1:
                    yield self.ports[i+1][j].agg_in_i.put((agg_out_i, target_index))
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i+1}, {j}) through (agg_out_i, agg_in_i), target_ijk={target_index}")
                else:
                    yield self.boundaries[0][j].agg_out.put(agg_out_i)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (0, {j}) agg_out through agg_out_i, target_ijk={target_index}")

            if use_agg_j:
                agg_out_j, target_index = yield self.ports[i][j].agg_out_j.get()
                if j != self.num_PEs[1]-1:
                    yield self.ports[i][j+1].agg_in_j.put((agg_out_j, target_index))
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i}, {j+1}) through (agg_out_j, agg_in_j), target_ijk={target_index}")
                else:
                    yield self.boundaries[1][i].agg_out.put(agg_out_j)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (1, {i}) agg_out through agg_out_j, target_ijk={target_index}")

