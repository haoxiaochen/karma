import simpy
import math
from MtxGen import get_num_halo_points
from loguru import logger

class HEU:
    def __init__(self, env, cfg, bufs, data, bd_ports, position):
        self.env = env
        self.cfg = cfg
        self.bufs = bufs
        self.data = data
        self.boundary_ports = bd_ports
        self.position = position

        self.num_PEs = cfg["Arch"]["NumPEs"]
        self.num_lanes = cfg["Arch"]["HaloLanes"][position]
        self.stencil_type = cfg["StencilType"]
        self.dims = cfg["NumDims"]
        self.halo_points = get_num_halo_points(self.stencil_type, self.dims, position, self.num_PEs[1] if position==0 else self.num_PEs[0])
        self.add_counter = 0
        # the number of processes is determined by the number of halo points
        self.actions = [env.process(self.run(i)) for i in range(self.halo_points)]

    def run(self, i):
        while True:
            tick = self.env.now
            yield self.env.process(self.bufs.halo_vec_in.access(1))
            # logger.trace(f"HEU({self.position}, {i}) is Waiting for b")
            b_val, b_idx = yield self.data.halo_vec_in[i].get()
            b_ijk = yield self.data.halo_idx_in[i].get()
            # logger.trace(f"HEU({self.position}, {i}) received b")

            x_ijk = 0
            out = 0; agg_out = 0
            out_flag = 0; agg_flag = 0
            if self.stencil_type == 0: # Star
                # logger.trace(f"HEU({self.position}, {i}) is Waiting for output")
                out, x_ijk = yield self.boundary_ports[i].out.get()
                # logger.trace(f"HEU({self.position}, {i}) received the output")
                self.add_counter += 1
                out_flag = 1

            elif self.stencil_type == 1: # Star
                if i % 2 == 0:
                    out = yield self.boundary_ports[i // 2].out.get()
                    out_flag = 1
                else:
                    out = yield self.boundary_ports[i // 2].agg_out.get()
                    agg_flag = 1
                self.add_counter += 1

            elif self.stencil_type == 2: # Diamond
                if self.position == 0: # HEU_X
                    out = yield self.boundary_ports[i].out.get() if i != self.num_PEs[1] else 0
                    agg_out = yield self.boundary_ports[i - 1].agg_out.get() if i != 0 else 0

                    out_flag = 1 if i != self.num_PEs[1] else 0
                    agg_flag = 1 if i != 0 else 0
                    self.add_counter += (agg_flag + out_flag)
                else: # HEU_Y
                    out = yield self.boundary_ports[i].out.get()
                    out_flag = 1
                    self.add_counter += 1

            elif self.stencil_type == 3: # Box
                if self.position == 0: # HEU_X
                    out = yield self.boundary_ports[i].out.get() if i != self.num_PEs[1] else 0
                    agg_out = yield self.boundary_ports[i - 1].agg_out.get() if i != 0 else 0

                    out_flag = 1 if i != self.num_PEs[1] else 0
                    agg_flag = 1 if i != 0 else 0
                    self.add_counter += (agg_flag + out_flag)
                else: # HEU_Y
                    if i % 2 == 0:
                        out = yield self.boundary_ports[i // 2].out.get()
                        out_flag = 1
                    else:
                        out = yield self.boundary_ports[i // 2].agg_out.get()
                        agg_flag = 1
                    self.add_counter += 1

            # 对于index[0] < 0的无效值，需要先取走agg & in端口值，避免阻塞
            new_b = b_val - (out + agg_out)
            if b_idx[0] < 0:
                yield self.env.process(self.bufs.halo_vec_out.access(1))
                logger.trace(f"(Cycle {self.env.now}) HEU ({self.position}, {i}) waiting for output")
                yield self.data.halo_vec_out[i].put((new_b, b_idx, out_flag, agg_flag))
                yield self.data.halo_idx_out[i].put(b_ijk)
                logger.trace(f"(Cycle {self.env.now}) HEU ({self.position}, {i}) consume {x_ijk} and ignore the invalid b{b_idx}")
                continue

            logger.trace(f"(Cycle {self.env.now}) HEU ({self.position}, {i}): get data ready takes {self.env.now - tick} cycles")
            logger.info(f"(Cycle {self.env.now}) HEU ({self.position}, {i}): update b{b_ijk}={new_b} with b={b_val}, out={out}, agg_out={agg_out}, x_ijk={x_ijk}")

            # process time is determined by the number of lanes
            times = math.ceil(self.halo_points / self.num_lanes)
            delay = self.cfg["Delay"]["Add"] * times
            yield self.env.timeout(delay)
            yield self.env.process(self.bufs.halo_vec_out.access(1))
            yield self.data.halo_vec_out[i].put((new_b, b_idx, out_flag, agg_flag))
            yield self.data.halo_idx_out[i].put(b_ijk)
            logger.trace(f"(Cycle {self.env.now}) HEU ({self.position}, {i}): update b{b_ijk}={new_b} one iteration takes {self.env.now - tick} cycles")
