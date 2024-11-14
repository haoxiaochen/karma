import json
import simpy
import sys
import os
from loguru import logger
from MtxGen import get_linear_system, preprocess
from Accelerator import *
import time

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <config_file> x y z")
        sys.exit(1)

    # log_dir = "./log"
    logger.remove()
    aggr_log_path = "log/aggr.log"
    scalar_log_path = "log/scalar.log"
    vec_log_path = "log/vec.log"
    heu_log_path = "log/heu.log"
    compute_log_path = "log/compute.log"

    if os.path.exists(aggr_log_path):
        os.remove(aggr_log_path)
    if os.path.exists(vec_log_path):
        os.remove(vec_log_path)
    if os.path.exists(scalar_log_path):
        os.remove(scalar_log_path)
    if os.path.exists(heu_log_path):
        os.remove(heu_log_path)
    if os.path.exists(compute_log_path):
        os.remove(compute_log_path)

    logger.add(aggr_log_path, format="<level>{message}</level>",
               level="TRACE", filter=lambda r: "Aggregator" in r["message"])
    logger.add(scalar_log_path, format="<level>{message}</level>",
               level="TRACE", filter=lambda r: "ScalarUnit" in r["message"])
    logger.add(vec_log_path, format="<level>{message}</level>",
               level="TRACE", filter=lambda r: "VectorUnit" in r["message"])
    logger.add(heu_log_path, format="<level>{message}</level>",
               level="TRACE", filter=lambda r: "HEU" in r["message"])
    logger.add(compute_log_path, format="<level>{message}</level>",
               level="TRACE", filter=lambda r: "PE(0, 0) ScalarUnit: compute variable" in r["message"])

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
    start_time = time.time()
    env.run(until=proc)
    print(f"\nSimulation Time: {time.time() - start_time:.2f}s")
    print("Correctness:", acc.check_correctness())
    acc.print()
