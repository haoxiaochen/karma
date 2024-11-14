import simpy
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
    def __init__(self, env, cfg, bufs, data, spmat_data, ports, trans_ijk, i, j):
        self.env = env
        self.cfg = cfg
        self.bufs = bufs
        self.data = data
        self.spmat_data = spmat_data
        self.ports = ports
        self.stencil_type = cfg["StencilType"]
        self.dims = cfg["NumDims"]
        self.trans_ijk = trans_ijk
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

        # self.Aggregator()

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

            if self.trans_ijk:
                in_j_ijk = (yield self.ports.in_j_ijk.get()) if self.j != 0 else 0
                in_i_ijk = (yield self.ports.in_i_ijk.get()) if self.i != 0 else 0

            in_j = (yield self.ports.in_j.get()) if self.j != 0 else 0
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get in_j ready takes {self.env.now - tick} cycles")

            in_i = (yield self.ports.in_i.get()) if self.i != 0 else 0
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get in_i ready takes {self.env.now - tick} cycles")

            R_k = (yield self.R_k.get()) if ijk_index[2] != 0 else 0
            logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: get all data ready takes {self.env.now - tick} cycles")

            sum = in_i + in_j + R_k
            x = (b - sum) / aii

            if self.trans_ijk:
                logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: compute variable x{ijk_index}={x} with aii={aii}, b={b}, in_i={in_i}, in_j={in_j}, R_k={R_k}, in_i_ijk={in_i_ijk}, in_j_ijk={in_j_ijk}")
            else:
                logger.info(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) ScalarUnit: compute variable x{ijk_index}={x} with aii={aii}, b={b}, in_i={in_i}, in_j={in_j}, R_k={R_k}")

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
        dim0_index = 0
        # stall = False
        # vec_A = []
        while True:
            tick = self.env.now
            if dim0_index < self.data.dim0_extent:
                new_x, ijk_index = yield self.new_x.get()
            else:
                new_x, ijk_index = 0, (0, 0, 0)

            yield self.env.process(self.bufs.domain_mtx.access(self.num_points))
            vec_A = yield self.spmat_data.domain_mtx[self.i][self.j].get()

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

                    z = self.data.size[2]
                    offset = self.z_offset[self.stencil_type][sche_id] - self.id2stage[sche_id]
                    z_target = dim0_index + offset
                    if z_target >= 0 and z_target < self.data.dim0_extent and \
                        (offset <= 0 or z_target % z != 0):
                        # 若z_target % z == 0表明target与当前不在同一个tile
                        yield self.vec_results[sche_id].put(vec_x[sche_id] * vec_A[sche_id])
                        sche_ids.append(sche_id)
                        mul_points += 1
                    else:
                        sche_ids.append(-1)

                logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: veclen={len(sche_ids)} terms={sche_ids} with shift_x={self.shift_x}, vec_A={vec_A}")

            self.mul_counter += mul_points
            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) VectorUnit: one iteration takes {self.env.now - tick} cycles")
            dim0_index += 1

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
        self.dim0_index = 0
        while True:
            tick = self.env.now
            self.agg_ijk_index = yield self.ijk_index.get()
            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: get data ready takes {self.env.now - tick} cycles")

            if self.dims == 3:
                # A_vec_target [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
                if self.stencil_type == 0: # Star7P
                    out_i_proc = self.env.process(self.buf(vec_results[2], out_i, 0))
                    out_j_proc = self.env.process(self.buf(vec_results[1], out_j, 0))
                    rk_proc = self.env.process(self.buf(vec_results[0], rk, 1))
                    yield out_i_proc & out_j_proc & rk_proc

                # A_vec_target [(0, 0, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, -1), (0, 2, -1)]
                elif self.stencil_type == 1: # Star13P
                    out_i_proc = self.env.process(self.adder([agg_in_i, vec_results[1]], out_i, 0))
                    out_j_proc = self.env.process(self.adder([agg_in_j, vec_results[2]], out_j, 0))
                    rk_proc = self.env.process(self.adder([vec_results[0], vec_results[3]], rk, 1))
                    agg_out_i_proc = self.env.process(self.buf(vec_results[4], agg_out_i, -1))
                    agg_out_j_proc = self.env.process(self.buf(vec_results[5], agg_out_j, -1))
                    yield out_i_proc & out_j_proc & rk_proc & agg_out_i_proc & agg_out_j_proc

                # A_vec_target [(0, 0, 1), (1, 0, 0), (0, 1, 0),
                #               (0, 1, 0), (1, 1, -1),
                #               (1, 1, -1)]
                elif self.stencil_type == 2: # diamond13P
                    out_i_proc = self.env.process(self.buf(vec_results[1], out_i, 0))
                    out_j_proc = self.env.process(self.adder([agg_in_i, vec_results[2], vec_results[3]], out_j, 0))
                    rk_proc = self.env.process(self.buf(vec_results[0], rk, 1))
                    agg_out_i_proc = self.env.process(self.adder([vec_results[4], vec_results[5]], agg_out_i, -1))
                    yield out_i_proc & out_j_proc & rk_proc & agg_out_i_proc

                # A_vec_target [(0, 0, 1), (1, 0, 0), (0, 1, 0),
                #               (0, 1, 0), (1, 0, 0),
                #               (0, 1, 0), (1, 1, -1), (1, 0, 0),
                #               (1, 1, -1),
                #               (1, 1, -1), (1, 2, -2),
                #               (1, 2, -2),
                #               (1, 2, -2)]
                elif self.stencil_type == 3: # box27P
                    out_i_proc = self.env.process(self.adder([vec_results[1], vec_results[4], vec_results[7]], out_i, 0))
                    out_j_proc = self.env.process(self.adder([agg_in_i, vec_results[2], vec_results[3], vec_results[5]], out_j, 0))
                    agg_out_i_proc = self.env.process(self.adder([agg_in_j, vec_results[6], vec_results[8], vec_results[9]], agg_out_i, -1))
                    agg_out_j_proc = self.env.process(self.adder([vec_results[10], vec_results[11], vec_results[12]], agg_out_j, -2))
                    rk_proc = self.env.process(self.buf(vec_results[0], rk, 1))
                    yield out_i_proc & out_j_proc & rk_proc & agg_out_i_proc & agg_out_j_proc
            else:
                pass

            logger.trace(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: one iteration takes {self.env.now - tick} cycles")
            self.dim0_index += 1

    # def Aggregator_new(self):
    #     # bind name to port for debugging
    #     agg_in_i = (self.ports.agg_in_i, "agg_in_i")
    #     agg_in_j = (self.ports.agg_in_j, "agg_in_j")
    #     agg_out_i = (self.ports.agg_out_i, "agg_out_i")
    #     agg_out_j = (self.ports.agg_out_j, "agg_out_j")
    #     out_i = (self.ports.out_i, "out_i")
    #     out_j = (self.ports.out_j, "out_j")
    #     rk = (self.R_k, "R_k")
    #     vec_results = [(self.vec_results[i], f"term[{i}]") for i in range(self.num_points)]

    #     # generate adder & buf processes
    #     # self.dim0_index = 0

    #     if self.dims == 3:
    #         if self.stencil_type == 0:
    #             self.actions += [
    #                 self.env.process(self.buf(vec_results[2], out_i, 0)),
    #                 self.env.process(self.buf(vec_results[1], out_j, 0)),
    #                 self.env.process(self.buf(vec_results[0], rk, 1))
    #             ]
    #         elif self.stencil_type == 1:
    #             self.actions += [
    #                 self.env.process(self.adder([agg_in_i, vec_results[1]], out_i, 0)),
    #                 self.env.process(self.adder([agg_in_j, vec_results[2]], out_j, 0)),
    #                 self.env.process(self.adder([vec_results[0], vec_results[3]], rk, 1)),
    #                 self.env.process(self.buf(vec_results[4], agg_out_i, -1)),
    #                 self.env.process(self.buf(vec_results[5], agg_out_j, -1))
    #             ]
    #         elif self.stencil_type == 2:
    #             self.actions += [
    #                 self.env.process(self.buf(vec_results[1], out_i, 0)),
    #                 self.env.process(self.adder([agg_in_i, vec_results[2], vec_results[3]], out_j, 0)),
    #                 self.env.process(self.buf(vec_results[0], rk, 1)),
    #                 self.env.process(self.adder([vec_results[4], vec_results[5]], agg_out_i, -1))
    #             ]
    #         elif self.stencil_type == 3:
    #             self.actions += [
    #                 self.env.process(self.adder([vec_results[1], vec_results[4], vec_results[7]], out_i, 0)),
    #                 self.env.process(self.adder([agg_in_i, vec_results[2], vec_results[3], vec_results[5]], out_j, 0)),
    #                 self.env.process(self.adder([agg_in_j, vec_results[6], vec_results[8], vec_results[9]], agg_out_i, -1)),
    #                 self.env.process(self.adder([vec_results[10], vec_results[11], vec_results[12]], agg_out_j, -2)),
    #                 self.env.process(self.buf(vec_results[0], rk, 1))
    #             ]
    #         # yield simpy.AllOf(self.env, actions)

    # inputs [(port, name)] output (port, name)
    def adder(self, inputs, output, z_target_offset):
        output_port, output_name = output
        z = self.data.size[2]
        z_target = self.dim0_index + z_target_offset

        if z_target_offset > 0 and \
             (z_target % z == 0 or \
             (z_target >= self.data.dim0_extent)): # 下传上，不在同一个tile则舍弃
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: index={self.agg_ijk_index}, useless output to {output_name} is ignored")
            return
        elif z_target < 0 or z_target >= self.data.dim0_extent: # 上传下，若目标坐标为负数，舍弃
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: z_target={z_target}, index={self.agg_ijk_index}, useless output to {output_name} is ignored")
            return

        sum = 0
        add_n = 0
        input_str = ""
        for in_port, in_name in inputs:
            if (in_name == "agg_in_i" and self.i == 0) or (in_name == "agg_in_j" and self.j == 0):
                continue
            input_data = yield in_port.get()
            sum += input_data
            input_str += (in_name + "+")
            add_n += 1

        if add_n >= 2:
            yield self.env.timeout(self.cfg['Delay']['Add'])
        logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: waiting for output port={output_name} to be available")
        yield output_port.put(sum)

        if self.trans_ijk:
            if output_name == "out_j":
                yield self.ports.out_j_ijk.put(self.agg_ijk_index)
            elif output_name == "out_i":
                yield self.ports.out_i_ijk.put(self.agg_ijk_index)

        self.add_counter += add_n - 1
        logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: add_n={add_n} agg_index={self.agg_ijk_index}, pass {input_str}={sum} through {output_name}")

    # def adder_new(self, inputs, output, z_target_offset):
    #     dim0_index = 0
    #     output_port, output_name = output
    #     z = self.data.size[2]
    #     while True:
    #         z_target = dim0_index + z_target_offset
    #         dim0_index += 1
    #         if z_target >= self.data.dim0_extent:
    #             return

    #         if z_target_offset > 0 and \
    #             (z_target % z == 0 or \
    #             (z_target >= self.data.dim0_extent)): # 下传上，不在同一个tile则舍弃
    #             logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: index={self.agg_ijk_index}, useless output to {output_name} is ignored")
    #             continue
    #         elif z_target < 0 or z_target >= self.data.dim0_extent: # 上传下，若目标坐标为负数，舍弃
    #             logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: z_target={z_target}, index={self.agg_ijk_index}, useless output to {output_name} is ignored")
    #             continue

    #         sum = 0
    #         add_n = 0
    #         input_str = ""
    #         for in_port, in_name in inputs:
    #             if (in_name == "agg_in_i" and self.i == 0) or (in_name == "agg_in_j" and self.j == 0):
    #                 continue
    #             input_data = yield in_port.get()
    #             sum += input_data
    #             input_str += (in_name + "+")
    #             add_n += 1

    #         if add_n >= 2:
    #             yield self.env.timeout(self.cfg['Delay']['Add'])
    #         logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: waiting for output port={output_name} to be available")
    #         yield output_port.put(sum)

    #         if self.trans_ijk:
    #             if output_name == "out_j":
    #                 yield self.ports.out_j_ijk.put(self.agg_ijk_index)
    #             elif output_name == "out_i":
    #                 yield self.ports.out_i_ijk.put(self.agg_ijk_index)

    #         self.add_counter += add_n - 1
    #         logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: add_n={add_n} agg_index={self.agg_ijk_index}, pass {input_str}={sum} through {output_name}")

    def buf(self, input, output, z_target_offset):
        input_port, input_name = input
        output_port, output_name = output
        z = self.data.size[2]

        z_target = self.dim0_index + z_target_offset
        if z_target_offset > 0 and \
            (z_target % z == 0 or \
             (z_target >= self.data.dim0_extent)): # 下传上，不在同一个tile则舍弃
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: index={self.agg_ijk_index}, useless output to {output_name} is ignored")
            return
        elif z_target < 0 or z_target >= self.data.dim0_extent: # 上传下，若目标坐标为负数，舍弃
            logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: index={self.agg_ijk_index}, useless output to {output_name} is ignored")
            return

        input_data = yield input_port.get()
        logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: waiting for output port={output_name} to be available")
        yield output_port.put(input_data)

        if self.trans_ijk:
            if output_name == "out_j":
                yield self.ports.out_j_ijk.put(self.agg_ijk_index)
            elif output_name == "out_i":
                yield self.ports.out_i_ijk.put(self.agg_ijk_index)

        logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) agg_index: {self.agg_ijk_index} Aggregator: pass {input_name}={input_data} through {output_name}")

    # def buf_new(self, input, output, z_target_offset):
    #     dim0_index = 0
    #     input_port, input_name = input
    #     output_port, output_name = output
    #     z = self.data.size[2]
    #     while True:
    #         z_target = dim0_index + z_target_offset
    #         dim0_index += 1
    #         if z_target >= self.data.dim0_extent:
    #             return
    #         if z_target_offset > 0 and \
    #             (z_target % z == 0 or \
    #             (z_target >= self.data.dim0_extent)): # 下传上，不在同一个tile则舍弃
    #             logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: index={self.agg_ijk_index}, useless output to {output_name} is ignored")
    #             continue
    #         elif z_target < 0 or z_target >= self.data.dim0_extent: # 上传下，若目标坐标为负数，舍弃
    #             logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: index={self.agg_ijk_index}, useless output to {output_name} is ignored")
    #             continue

    #         input_data = yield input_port.get()
    #         logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) Aggregator: waiting for output port={output_name} to be available")
    #         yield output_port.put(input_data)

    #         if self.trans_ijk:
    #             if output_name == "out_j":
    #                 yield self.ports.out_j_ijk.put(self.agg_ijk_index)
    #             elif output_name == "out_i":
    #                 yield self.ports.out_i_ijk.put(self.agg_ijk_index)

    #         logger.debug(f"(Cycle {self.env.now}) PE({self.i}, {self.j}) agg_index: {self.agg_ijk_index} Aggregator: pass {input_name}={input_data} through {output_name}")


class PEArray:
    def __init__(self, env, cfg, bufs, data, spmat_data, boundaries):
        self.env = env
        self.cfg = cfg
        self.data = data
        self.spmat_data = spmat_data
        self.boundaries = boundaries
        self.trans_ijk = False
        self.num_PEs = cfg["Arch"]["NumPEs"]
        self.ports = [[PEPorts(env) for _ in range(self.num_PEs[1])] for _ in range(self.num_PEs[0])]
        self.PEs = [[PE(env, cfg, bufs, data, spmat_data, self.ports[i][j], self.trans_ijk, i, j) for j in range(self.num_PEs[1])] for i in range(self.num_PEs[0])]

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
            out_i = yield self.ports[i][j].out_i.get()
            if self.trans_ijk:
                out_i_ijk = yield self.ports[i][j].out_i_ijk.get()

            if i != self.num_PEs[0]-1:
                yield self.ports[i+1][j].in_i.put(out_i)
                if self.trans_ijk:
                    yield self.ports[i + 1][j].in_i_ijk.put(out_i_ijk)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i+1}, {j}) through (out_i, in_i), out_i_ijk={out_i_ijk}")
                else:
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i+1}, {j}) through (out_i, in_i)")
            else:
                yield self.boundaries[0][j].out.put(out_i)
                if self.trans_ijk:
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (0, {j}) through out_i, out_i_ijk={out_i_ijk}")
                else:
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (0, {j}) through out_i")

            # trans j
            if self.trans_ijk:
                out_j_ijk = yield self.ports[i][j].out_j_ijk.get()
            out_j = yield self.ports[i][j].out_j.get()

            if j != self.num_PEs[1]-1:
                yield self.ports[i][j+1].in_j.put(out_j)
                if self.trans_ijk:
                    yield self.ports[i][j + 1].in_j_ijk.put(out_j_ijk)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i}, {j+1}) through (out_j, in_j), out_j_ijk={out_j_ijk}")
                else:
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i}, {j+1}) through (out_j, in_j)")
            else:
                yield self.boundaries[1][i].out.put(out_j)
                if self.trans_ijk:
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (1, {i}) through out_j, out_j_ijk={out_j_ijk}")
                else:
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (1, {i}) through out_j")


    def trans_aggr(self, i, j):
        stencil_type = self.cfg["StencilType"]
        use_agg_i = False if stencil_type == 0 else True    # Star & Diamond & Box
        use_agg_j = True if stencil_type == 1 or stencil_type == 3 else False    # Star & Box
        if not use_agg_i and not use_agg_j:
            return
        while True:
            if use_agg_i:
                agg_out_i = yield self.ports[i][j].agg_out_i.get()
                if i != self.num_PEs[0]-1:
                    yield self.ports[i+1][j].agg_in_i.put(agg_out_i)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i+1}, {j}) through (agg_out_i, agg_in_i)")
                else:
                    yield self.boundaries[0][j].agg_out.put(agg_out_i)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (0, {j}) through agg_out_i")

            if use_agg_j:
                agg_out_j = yield self.ports[i][j].agg_out_j.get()
                if j != self.num_PEs[1]-1:
                    yield self.ports[i][j+1].agg_in_j.put(agg_out_j)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to PE({i}, {j+1}) through (agg_out_j, agg_in_j)")
                else:
                    yield self.boundaries[1][i].agg_out.put(agg_out_j)
                    logger.trace(f"(Cycle {self.env.now}) PEArray: pass from PE({i}, {j}) to HEU (1, {i}) through agg_out_j")

