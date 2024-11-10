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
            b_val, b_idx = yield self.data.halo_vec_in[i].get()
            b_ijk = yield self.data.halo_idx_in[i].get()

            out = 0; agg_out = 0
            if self.stencil_type == 0: # Star
                out = yield self.boundary_ports[i].out.get()
            elif self.stencil_type == 2: # Diamond
                if self.position == 0:
                    out = yield self.boundary_ports[i].out.get() if i != self.num_PEs[1] else 0
                    agg_out = yield self.boundary_ports[i].agg_out.get()
                else:
                    out = yield self.boundary_ports[i].out.get()
            logger.trace(f"(Cycle {self.env.now}) HEU ({self.position}, {i}): get data ready takes {self.env.now - tick} cycles")

            new_b = b_val - (out + agg_out)
            self.add_counter += 2
            logger.info(f"(Cycle {self.env.now}) HEU ({self.position}, {i}): update b{b_ijk}={new_b} with b={b_val}, out={out}, agg_out={agg_out}")

            # process time is determined by the number of lanes
            times = math.ceil(self.halo_points / self.num_lanes)
            delay = self.cfg["Delay"]["Add"] * times
            yield self.env.timeout(math.ceil(delay))
            yield self.env.process(self.bufs.halo_vec_out.access(1))
            yield self.data.halo_vec_out[i].put((new_b, b_idx))
            yield self.data.halo_idx_out[i].put(b_ijk)
            logger.trace(f"(Cycle {self.env.now}) HEU ({self.position}, {i}): one iteration takes {self.env.now - tick} cycles")
