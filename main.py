import json
import simpy
import sys
from loguru import logger
from MtxGen import get_linear_system, preprocess
from Accelerator import *

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <config_file> x y z")
        sys.exit(1)
    logger.remove()
    logger.add(sys.stderr, format="<level>{message}</level>", level="INFO", filter=lambda r: "" in r["message"])

    config_file = sys.argv[1]
    config = read_config(config_file)
    dims = config["NumDims"]
    stencil = config["StencilType"]
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    if dims == 3:
        z = int(sys.argv[4])
    else:
        z = 1
    data = get_linear_system(dims, stencil, x, y, z)
    data = preprocess(data, config["Arch"]["NumPEs"][0], config["Arch"]["NumPEs"][1], stencil)

    env = simpy.Environment()
    acc = Accelerator(env, config, data)
    proc = env.process(acc.wait_for_finish())
    env.run(until=proc)
    acc.print()
