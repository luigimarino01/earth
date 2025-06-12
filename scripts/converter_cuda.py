import numpy as np
from PIL import Image
from numba import cuda
from pathlib import Path

@cuda.jit
def encode_kernel_coords(lat_arr, lon_arr, out_arr):
    j, k = cuda.grid(2)
    if j < lat_arr.shape[0] and k < lon_arr.shape[0]:
        lat = lat_arr[j]
        lon = lon_arr[k]
        # Replace below with your actual encode logic
        out_arr[j, k, 0] = int(((lat + 90) / 180) * 255)  # Example mapping
        out_arr[j, k, 1] = int(((lon + 180) / 360) * 255)
        out_arr[j, k, 2] = 0
        out_arr[j, k, 3] = 255

@cuda.jit
def encode_kernel_uv(u_arr, v_arr, out_arr):
    j, k = cuda.grid(2)
    if j < u_arr.shape[0] and k < u_arr.shape[1]:
        u = u_arr[j, k]
        v = v_arr[j, k]
        # Replace below with your actual encode logic
        out_arr[j, k, 0] = int(((u + 50) / 100) * 255)
        out_arr[j, k, 1] = int(((v + 50) / 100) * 255)
        out_arr[j, k, 2] = 0
        out_arr[j, k, 3] = 255

def netcdf_to_png(input_dir, filename, output_dir):
    lats, lons, u, v = load_data(input_dir, filename)
    output_folder = Path(output_dir) / filename
    output_folder.mkdir(exist_ok=True, parents=True)

    lat_arr = cuda.to_device(lats)
    lon_arr = cuda.to_device(lons)

    for i in range(u.shape[0]):
        u_frame = u[i]
        v_frame = v[i]
        u_device = cuda.to_device(u_frame)
        v_device = cuda.to_device(v_frame)

        coords_img_data = np.zeros((u.shape[1], u.shape[2], 4), dtype=np.uint8)
        uv_img_data = np.zeros((u.shape[1], u.shape[2], 4), dtype=np.uint8)

        coords_device = cuda.to_device(coords_img_data)
        uv_device = cuda.to_device(uv_img_data)

        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(u.shape[1] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(u.shape[2] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        encode_kernel_coords[blockspergrid, threadsperblock](lat_arr, lon_arr, coords_device)
        encode_kernel_uv[blockspergrid, threadsperblock](u_device, v_device, uv_device)

        coords_device.copy_to_host(coords_img_data)
        uv_device.copy_to_host(uv_img_data)

        Image.fromarray(coords_img_data, mode="RGBA").save(output_folder / f"{i}.coords.png")
        Image.fromarray(uv_img_data, mode="RGBA").save(output_folder / f"{i}.uv.png")