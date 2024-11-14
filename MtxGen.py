# 读入命令行参数, 4-5个数字, 代表stencil类型, 维数, x, y, z

import sys
import numpy as np
import math
import random

def get_num_domain_points(stencil_type, dim):
    if stencil_type == 0:
        if dim == 2: return 2
        if dim == 3: return 3
    if stencil_type == 1:
        if dim == 2: return 4
        if dim == 3: return 6
    if stencil_type == 2:
        if dim == 2: return 3
        if dim == 3: return 6
    if stencil_type == 3:
        if dim == 2: return 4
        if dim == 3: return 13

def get_num_halo_points(stencil_type, dim, position, base):
    if stencil_type == 0: return base
    if stencil_type == 1: return 2*base
    if stencil_type == 2:
        return base+1 if position == 0 else base
    if stencil_type == 3:
        return base+1 if position == 0 else 2*base

def get_affine_stencil_points(dim, stencil_type):
    if stencil_type == 0:
        # 3D-Star-7P/2D-Star-5P
        if dim == 2:
            stencil_points = [
                (0, -1), (-1, 0),
            ]
        elif dim == 3:
            stencil_points = [
                (0, 0, -1), (0, -1, 0), (-1, 0, 0),
            ]
    elif stencil_type == 1:
        # 3D-Star-13P/2D-Star-9P
        if dim == 2:
            stencil_points = [
                (-1, 0), (0, -1),
                (-2, 0), (0, -2),
            ]
        elif dim == 3:
            stencil_points = [
                (0, 0, -1), (-1, 0, 0), (0, -1, 0),
                (0, 0, -2), (-2, 0, 0), (0, -2, 0),
            ]
    elif stencil_type == 2:
        # 3D-Diamond-13P / 2D-Diamond-7P
        if dim == 2:
            stencil_points = [
                (0, -1),
                (-1, 0),
                (-1, 1),
            ]
        elif dim == 3:
            stencil_points = [
                (0, 0, -1), (-1, 0, 0), (0, -1, 0),
                (0, -1, -1), (-1, -1, 0),
                (-1, -1, -1),
            ]
    elif stencil_type == 3:
        # 3D-Box-27P / 2D-Box-9P
        if dim == 2:
            stencil_points = [
                (-1, -1),
                (0, -1),
                (-1, 0),
                (-1, 1),
            ]
        elif dim == 3:
            stencil_points = [
                (0, 0, -1), (-1, 0, 0), (0, -1, 0),
                (0, -1, -1), (-1, 0, -1),
                (0, -1, -2), (-1, -1, -1), (-1, 0, -2),
                (-1, -1, -2),
                (-1, -1, -3), (-1, -2, -2),
                (-1, -2, -3),
                (-1, -2, -4),
            ]
    return stencil_points

def get_stencil_points(dim, stencil_type):
    if stencil_type == 0:
        # 3D-Star-7P/2D-Star-5P
        if dim == 2:
            stencil_points = [
                (0, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (0, 1),
            ]
        elif dim == 3:
            stencil_points = [
                (0, 0, -1),
                (0, -1, 0),
                (-1, 0, 0),

                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1),
            ]
    elif stencil_type == 1:
        # 3D-Star-13P/2D-Star-9P
        if dim == 2:
            stencil_points = [
                (0, -2),
                (0, -1),
                (-2, 0),
                (-1, 0),
                (0, 0),
                (1, 0),
                (2, 0),
                (0, 1),
                (0, 2),
            ]
        elif dim == 3:
            stencil_points = [
                (0, 0, -1),
                (-1, 0, 0),
                (0, -1, 0),

                (0, 0, -2),
                (-2, 0, 0),
                (0, -2, 0),

                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (0, 1, 0),
                (0, 2, 0),
                (0, 0, 1),
                (0, 0, 2),
            ]
    elif stencil_type == 2:
        # 3D-Diamond-13P / 2D-Diamond-7P
        if dim == 2:
            stencil_points = [
                (0, -1),
                (1, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
            ]
        elif dim == 3:
            stencil_points = [
                (0, 0, -1),
                (-1, 0, 0),
                (0, -1, 0),
                (-1, 1, 0),
                (-1, 0, 1),
                (0, -1, 1),


                (0, 0, 0),
                (1, 0, 0),
                (1, -1, 0),
                (0, 1, 0),
                (0, 0, 1),
                (1, 0, -1),
                (0, 1, -1),
            ]
    elif stencil_type == 3:
        # 3D-Box-27P / 2D-Box-9P
        if dim == 2:
            stencil_points = [
                (-1, -1),
                (0, -1),
                (1, -1),
                (-1, 0),
                (0, 0),
                (1, 0),
                (-1, 1),
                (0, 1),
                (1, 1),
            ]
        elif dim == 3:
            stencil_points = [
                (-1, -1, -1),
                (0, -1, -1),
                (1, -1, -1),
                (-1, 0, -1),
                (0, 0, -1),
                (1, 0, -1),
                (-1, 1, -1),
                (0, 1, -1),
                (1, 1, -1),
                (-1, -1, 0),
                (0, -1, 0),
                (1, -1, 0),
                (-1, 0, 0),
                (0, 0, 0),
                (1, 0, 0),
                (-1, 1, 0),
                (0, 1, 0),
                (1, 1, 0),
                (-1, -1, 1),
                (0, -1, 1),
                (1, -1, 1),
                (-1, 0, 1),
                (0, 0, 1),
                (1, 0, 1),
                (-1, 1, 1),
                (0, 1, 1),
                (1, 1, 1),
            ]
    return stencil_points


def get_linear_system(dim, stencil_type, x, y, z):
    grid_size = x * y * z
    stencil_points = get_affine_stencil_points(dim, stencil_type)
    stencil_length = len(stencil_points)
    matrix_value = np.zeros((grid_size, stencil_length))
    matrix_diag = np.zeros(grid_size)
    right_hand_side = np.ones(grid_size)

    for k in range(x):
        for j in range(y):
            for i in range(z):
                # only lower triangular
                sum = 0
                idx = k * y * z + j * z + i
                for l in range(stencil_length):
                    if dim == 2:
                        dx, dy = stencil_points[l]
                        dz = 0
                    else:
                        dx, dy, dz = stencil_points[l]
                    x_new = k + dx
                    y_new = j + dy
                    z_new = i + dz
                    idx_new = x_new * y * z + y_new * z + z_new
                    if x_new >= 0 and x_new < x and \
                        y_new >= 0 and y_new < y and \
                        z_new >= 0 and z_new < z:

                        rand_num = -random.randint(1, 10)
                        # rand_num = -1
                        matrix_value[idx_new][l] = rand_num
                        sum += rand_num
                    else:
                        matrix_value[idx_new][l] = 0.0
                matrix_diag[idx] = -sum + 1.0

    data = { "size": (x, y, z), "A": matrix_value, "diag_a": matrix_diag, "b": right_hand_side }
    return data


def processPGC(size, data_A, tile_x, tile_y, stencil_type, dims):
    x, y, z = size
    n = x * y * z
    stencil_stages_3d = [1, 2, 3, 7]
    num_tile_x = math.ceil(x / tile_x)
    num_tile_y = math.ceil(y / tile_y)
    num_tiles = num_tile_x * num_tile_y
    stencil_length = data_A.shape[1]
    dim_shape_A = (num_tiles * z + stencil_stages_3d[stencil_type] - 1, tile_x, tile_y, stencil_length)
    matrix_value = np.zeros(dim_shape_A)
    stencil_id2stage_3d = [
        [0, 0, 0],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 2],
        [0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 5, 6]
    ]
    id2stage = stencil_id2stage_3d[stencil_type]
    if dims == 3:
        for out_i in range(num_tile_x):
            for out_j in range(num_tile_y):
                for k in range(z):
                    tile_idx = out_i * num_tile_y + out_j
                    dim_0 = tile_idx * z + k
                    for in_i in range(tile_x):
                        for in_j in range(tile_y):
                            total_i = out_i * tile_x + in_i
                            total_j = out_j * tile_y + in_j
                            addr = total_i * y * z + total_j * z + k
                            if addr < n:
                                for l in range(stencil_length):
                                    matrix_value[dim_0 + id2stage[l]][in_i][in_j][l] = data_A[addr][l]
    return matrix_value

def preprocess(data, tile_x, tile_y, stencil_type, dims):
    x, y, z = data["size"]
    n = x * y * z
    num_tile_x = math.ceil(x / tile_x)
    num_tile_y = math.ceil(y / tile_y)
    num_tiles = num_tile_x * num_tile_y
    # store the tiling result in a 3D tensor
    dim_shape = (num_tiles*z, tile_x, tile_y)
    matrix_diag = np.zeros(dim_shape)
    right_hand_side = np.ones(dim_shape)
    vec_index = np.zeros(dim_shape, dtype=object)
    # padding for halo data
    padd_x = get_num_halo_points(stencil_type, dims, 0, tile_y)
    padd_y = get_num_halo_points(stencil_type, dims, 1, tile_x)
    halo_x = np.zeros((num_tiles*z, padd_x), dtype=object)
    halo_y = np.zeros((num_tiles*z, padd_y), dtype=object)
    b_valid = np.zeros(dim_shape, dtype=np.int8)

    matrix_value = processPGC(data["size"], data['A'], tile_x, tile_y, stencil_type, dims)
    # domain data
    for out_i in range(num_tile_x):
        for out_j in range(num_tile_y):
            for k in range(z):
                tile_idx = out_i * num_tile_y + out_j
                dim_0 = tile_idx * z + k
                for in_i in range(tile_x):
                    for in_j in range(tile_y):
                        total_i = out_i * tile_x + in_i
                        total_j = out_j * tile_y + in_j
                        addr = total_i * y * z + total_j * z + k
                        if addr < n:
                            matrix_diag[dim_0][in_i][in_j] = data["diag_a"][addr]
                            right_hand_side[dim_0][in_i][in_j] = data["b"][addr]
                        else:
                            matrix_diag[dim_0][in_i][in_j] = 1
                            right_hand_side[dim_0][in_i][in_j] = 0
                        vec_index[dim_0][in_i][in_j] = (total_i, total_j, k)
    # halo data
    for out_i in range(num_tile_x):
        for out_j in range(num_tile_y):
            for k in range(z):
                tile_idx = out_i * num_tile_y + out_j
                dim_0 = tile_idx * z + k
                if stencil_type == 0: # Star7P
                    # Star7P: padd_x = padd_y = base
                    for p in range(padd_x):
                        halo_tile_x = out_i + 1
                        halo_tile_y = out_j if p < tile_y else out_j + 1
                        halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                        halo_dim_0 = halo_tile_idx * z + k if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y else -1
                        halo_x[dim_0][p] = (halo_dim_0, 0, p % tile_y)

                        if halo_dim_0 >= 0:
                            b_valid[halo_x[dim_0][p]] += (1 << 1) # out_i

                    for p in range(padd_y):
                        halo_tile_x = out_i
                        halo_tile_y = out_j + 1
                        halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                        halo_dim_0 = halo_tile_idx * z + k if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y else -1
                        halo_y[dim_0][p] = (halo_dim_0, p % tile_x, 0)

                        if halo_dim_0 >= 0:
                            b_valid[halo_y[dim_0][p]] += 1 # out_j

                elif stencil_type == 1: # Star13P
                    # in和agg交替映射 halo_x[0]->in, halo_x[1]->agg, ...
                    # padd_x = padd_y = 2 * base
                    for p in range(padd_x):
                        halo_tile_x = out_i + 1
                        halo_tile_y = out_j
                        halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                        halo_dim_0 = halo_tile_idx * z + k if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y else -1
                        halo_x[dim_0][p] = (halo_dim_0, p % 2, (p // 2) % tile_y)

                        if halo_dim_0 >= 0:
                            if p % 2 == 0:
                                b_valid[halo_x[dim_0][p]] += (1 << 1) # out_i
                            else:
                                b_valid[halo_x[dim_0][p]] += (1 << 2) # agg_i

                    for p in range(padd_y):
                        halo_tile_x = out_i
                        halo_tile_y = out_j + 1
                        halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                        halo_dim_0 = halo_tile_idx * z + k if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y else -1
                        halo_y[dim_0][p] = (halo_dim_0, (p // 2) % tile_x, p % 2)

                        if halo_dim_0 >= 0:
                            if p % 2 == 0:
                                b_valid[halo_y[dim_0][p]] += 1 # out_j
                            else:
                                b_valid[halo_y[dim_0][p]] += (1 << 3) # agg_j

                elif stencil_type == 2: # diamond13P
                    # Diamond13P: padd_x = base + 1, padd_y = base
                    for p in range(padd_x):
                        halo_tile_x = out_i + 1
                        halo_tile_y = out_j if p < tile_y else out_j + 1
                        halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                        halo_dim_0 = halo_tile_idx * z + k if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y else -1
                        halo_x[dim_0][p] = (halo_dim_0, 0, p % tile_y)

                        if halo_dim_0 >= 0:
                            if p == 0: # out_i
                                b_valid[halo_x[dim_0][p]] += (1 << 1)
                            elif p == tile_y: # agg_i
                                b_valid[halo_x[dim_0][p]] += (1 << 2)
                            else: # out_i & agg_i
                                b_valid[halo_x[dim_0][p]] += ((1 << 2) + (1 << 1))

                    for p in range(padd_y):
                        halo_tile_x = out_i
                        halo_tile_y = out_j + 1
                        halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                        halo_dim_0 = halo_tile_idx * z + k if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y else -1
                        halo_y[dim_0][p] = (halo_dim_0, p % tile_x, 0)

                        if halo_dim_0 >= 0:
                            b_valid[halo_y[dim_0][p]] += 1 # out_j

                elif stencil_type == 3: # Box27P
                    # in和agg交替映射 halo_y[0]->in, halo_y[1]->agg, ...
                    # padd_x = base + 1, padd_y = 2 * base
                    for p in range(padd_x):
                        halo_tile_x = out_i + 1
                        halo_tile_y = out_j if p < tile_y else out_j + 1
                        halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                        halo_dim_0 = halo_tile_idx * z + k if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y else -1
                        halo_x[dim_0][p] = (halo_dim_0, 0, p % tile_y)

                        if halo_dim_0 >= 0:
                            if p == 0: # out_i
                                b_valid[halo_x[dim_0][p]] += (1 << 1)
                            elif p == tile_y: # agg_i
                                b_valid[halo_x[dim_0][p]] += (1 << 2)
                            else: # out_i & agg_i
                                b_valid[halo_x[dim_0][p]] += ((1 << 2) + (1 << 1))

                    for p in range(padd_y):
                        halo_tile_x = out_i if p < 2 * tile_x - 1 else out_i + 1
                        halo_tile_y = out_j + 1
                        halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                        halo_dim_0 = halo_tile_idx * z + k if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y else -1
                        halo_y[dim_0][p] = (halo_dim_0, ((p + 1) // 2) % tile_x, p % 2)

                        if halo_dim_0 >= 0:
                            if p % 2 == 0:
                                b_valid[halo_y[dim_0][p]] += 1 # out_j
                            else:
                                b_valid[halo_y[dim_0][p]] += (1 << 3) # agg_j

    data = { "size": (x, y, z), "A": matrix_value, "diag_A": matrix_diag, "b": right_hand_side,
                                "b_valid": b_valid, "ijk": vec_index,
                                "halo_x": halo_x, "halo_y": halo_y, "x": np.zeros((x,y,z), dtype=object) }
    return data