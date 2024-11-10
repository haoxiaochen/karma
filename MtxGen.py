# 读入命令行参数, 4-5个数字, 代表stencil类型, 维数, x, y, z

import sys
import numpy as np
import math

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
        return 2*(base+1) if position == 0 else 2*base

def get_linear_system(dim, stencil_type, x, y, z):
    grid_size = x * y * z
    stencil_points = []

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
                (0, 0, -2),
                (0, 0, -1),
                (0, -2, 0),
                (0, -1, 0),
                (-2, 0, 0),
                (-1, 0, 0),
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
                (1, 0, -1),
                (0, 1, -1),
                (0, -1, 0),
                (1, -1, 0),
                (-1, 0, 0),
                (0, 0, 0),
                (1, 0, 0),
                (-1, 1, 0),
                (0, 1, 0),
                (0, -1, 1),
                (-1, 0, 1),
                (0, 0, 1),
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
    stencil_length = len(stencil_points)
    matrix_value = np.zeros((grid_size, stencil_length // 2))
    matrix_diag = np.zeros(grid_size)
    right_hand_side = np.ones(grid_size)

    for i in range(z):
        for j in range(y):
            for k in range(x):
                # only lower triangular
                cnt = 0
                for l in range(stencil_length // 2):
                    if dim == 2:
                        dx, dy = stencil_points[l]
                        dz = 0
                    else:
                        dx, dy, dz = stencil_points[l]
                    x_new = k + dx
                    y_new = j + dy
                    z_new = i + dz
                    if (
                        x_new >= 0
                        and x_new < x
                        and y_new >= 0
                        and y_new < y
                        and z_new >= 0
                        and z_new < z
                    ):
                        matrix_value[k * y * z + j * z + i][l] = -1.0
                        cnt += 1
                    else:
                        matrix_value[k * y * z + j * z + i][l] = 0.0
                matrix_diag[k * y * z + j * z + i] = cnt + 1.0
    data = { "size": (x, y, z), "A": matrix_value, "diag_a": matrix_diag, "b": right_hand_side }
    return data

def preprocess(data, tile_x, tile_y, stencil_type):
    x, y, z = data["size"]
    num_tile_x = math.ceil(x / tile_x)
    num_tile_y = math.ceil(y / tile_y)
    num_tiles = num_tile_x * num_tile_y
    # store the tiling result in a 3D tensor
    dim_shape = (num_tiles*z, tile_x, tile_y)
    stencil_length = len(data["A"][0])
    matrix_value = np.zeros(dim_shape + (stencil_length,))
    matrix_diag = np.zeros(dim_shape)
    right_hand_side = np.ones(dim_shape)
    vec_index = np.zeros(dim_shape, dtype=object)
    # padding for halo data
    halo_layers = 1
    padd_x = tile_y + (1 if stencil_type == 1 else 0)
    padd_y = tile_x
    halo_x = np.zeros((num_tiles*z, padd_x*halo_layers), dtype=object)
    halo_y = np.zeros((num_tiles*z, padd_y*halo_layers), dtype=object)
    halo_x_index = np.zeros((num_tiles*z, padd_x*halo_layers), dtype=object)
    halo_y_index = np.zeros((num_tiles*z, padd_y*halo_layers), dtype=object)
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
                        matrix_value[dim_0][in_i][in_j] = data["A"][addr]
                        matrix_diag[dim_0][in_i][in_j] = data["diag_a"][addr]
                        right_hand_side[dim_0][in_i][in_j] = data["b"][addr]
                        vec_index[dim_0][in_i][in_j] = (total_i, total_j, k)
    # halo data
    for out_i in range(num_tile_x):
        for out_j in range(num_tile_y):
            for k in range(z):
                tile_idx = out_i * num_tile_y + out_j
                dim_0 = tile_idx * z + k
                for p in range(padd_x):
                    halo_tile_x = out_i + 1
                    halo_tile_y = out_j if p < tile_y else out_j + 1
                    halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                    halo_dim_0 = halo_tile_idx * z + k if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y else -1
                    value = right_hand_side[halo_dim_0][0][p % tile_y] if halo_dim_0 >= 0 else 0
                    index = (halo_dim_0, 0, p % tile_y)
                    halo_x[dim_0][p % tile_y] = (value, index)
                    halo_x_index[dim_0][p % tile_y] = vec_index[halo_dim_0][0][p % tile_y] if halo_dim_0 >= 0 else (-1, -1, -1)
                for p in range(padd_y):
                    halo_tile_x = out_i if p < tile_x else out_i + 1
                    halo_tile_y = out_j + 1
                    halo_tile_idx = halo_tile_x * num_tile_y + halo_tile_y
                    halo_dim_0 = halo_tile_idx * z + k if halo_tile_x < num_tile_x and halo_tile_y < num_tile_y else -1
                    value = right_hand_side[halo_dim_0][p % tile_x][0] if halo_dim_0 >= 0 else 0
                    index = (halo_dim_0, 0, p % tile_x)
                    halo_y[dim_0][p % tile_x] = (value, index)
                    halo_y_index[dim_0][p % tile_x] = vec_index[halo_dim_0][p % tile_x][0] if halo_dim_0 >= 0 else (-1, -1, -1)

    data = { "size": (x, y, z), "A": matrix_value, "diag_A": matrix_diag, "b": right_hand_side, "ijk": vec_index, "x": np.zeros(dim_shape, dtype=object),
                                "halo_x": halo_x, "halo_x_ijk": halo_x_index, "halo_y": halo_y, "halo_y_ijk": halo_y_index }
    return data