import json
import simpy
import sys
import os
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

    log_path = "log/heu.log"
    if os.path.exists(log_path):
        os.remove(log_path)

    logger.remove()

    # logger.add(log_path, level="DEBUG")
    logger.add(log_path, format="<level>{message}</level>",
               level="TRACE", filter=lambda r: "(0, 4, 2)" in r["message"])

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
    data = preprocess(data, config["Arch"]["NumPEs"][0], config["Arch"]["NumPEs"][1], stencil, dims)

    env = simpy.Environment()
    acc = Accelerator(env, config, data, progressbar=True)
    proc = env.process(acc.wait_for_finish())
    env.run(until=proc)
    print("\nCorrectness:", acc.check_correctness())
    acc.print()
