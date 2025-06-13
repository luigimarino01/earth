import xarray as xr
import numpy as np
from PIL import Image
from pathlib import Path
from numba import cuda
import math

import time

VAR_U = "u"
VAR_V = "v"
VAR_LAT = "latitude"
VAR_LON = "longitude"

INPUT_DIR = Path("netcdf")
OUTPUT_DIR = Path("tmp")

def load_data(input_dir, filename):
    # Loads the geographical coordinates and the U and V components from the given NetCDF file
    ds = xr.open_dataset(input_dir.joinpath(filename + ".nc"))
    lats = ds[VAR_LAT].values
    lons = ds[VAR_LON].values
    u = ds[VAR_U].squeeze().values
    v = ds[VAR_V].squeeze().values
    return lats, lons, u, v

def float_to_uint16(value, min_val, max_val):
    if math.isnan(value):
        value = 0
    
    # Converts a 32-bit floating point value into a 16-bit unsigned integer
    norm = (value - min_val) / (max_val - min_val)
    norm = min(1, max(0, norm))
    return int(norm * 65535)

float_to_uint16_gpu = cuda.jit(device=True)(float_to_uint16)

def encode(pixel, val1, min1, max1, val2, min2, max2, cast=float_to_uint16):
    # Encodes two 32-bit floating point values into RGBA channels
    z1 = cast(val1, min1, max1)
    z2 = cast(val2, min2, max2)
    pixel[0] = (z1 >> 8) & 0xFF
    pixel[1] = z1 & 0xFF
    pixel[2] = (z2 >> 8) & 0xFF
    pixel[3] = z2 & 0xFF

encode_gpu = cuda.jit(device=True)(encode)

def encode_coords(lats, lons, img_data):
    for i in range(lats.shape[0]):
        for j in range(lons.shape[0]):
            encode(img_data[i, j], lats[i], -90.0, 90.0, lons[j], -180.0, 180.0)

@cuda.jit
def encode_kernel_coords(lats, lons, img_data):
    i, j = cuda.grid(2)
    if i < lats.shape[0] and j < lons.shape[0]:
        encode_gpu(img_data[i, j], lats[i], -90.0, 90.0, lons[j], -180.0, 180.0, float_to_uint16_gpu)

def encode_uv(u, v, img_data):
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            encode(img_data[i, j], u[i, j], -50.0, 50.0, v[i, j], -50.0, 50.0)

@cuda.jit
def encode_kernel_uv(u, v, img_data):
    i, j = cuda.grid(2)
    if i < u.shape[0] and j < u.shape[1]:
        encode_gpu(img_data[i, j], u[i, j], -50.0, 50.0, v[i, j], -50.0, 50.0, float_to_uint16_gpu)

def netcdf_to_png(input_dir, filename, output_folder, use_cuda=True):
    # Carica dati
    lats, lons, u, v = load_data(input_dir, filename)
    use_cuda = use_cuda and cuda.is_available()
    if use_cuda:
        # Copia lat/lon su GPU una volta sola
        lats_device = cuda.to_device(lats)
        lons_device = cuda.to_device(lons)

        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(u.shape[1] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(u.shape[2] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

    for i in range(u.shape[0]):
        start = time.time()
        coords_img_data = np.zeros((u.shape[1], u.shape[2], 4), dtype=np.uint8)
        uv_img_data = np.zeros((u.shape[1], u.shape[2], 4), dtype=np.uint8)

        if use_cuda:
            u_device = cuda.to_device(u[i])
            v_device = cuda.to_device(v[i])

            coords_device = cuda.to_device(coords_img_data)
            uv_device = cuda.to_device(uv_img_data)

            encode_kernel_coords[blockspergrid, threadsperblock](lats_device, lons_device, coords_device)
            encode_kernel_uv[blockspergrid, threadsperblock](u_device, v_device, uv_device)

            coords_device.copy_to_host(coords_img_data)
            uv_device.copy_to_host(uv_img_data)
        else:
            encode_coords(lats, lons, coords_img_data)
            encode_uv(u[i], v[i], uv_img_data)

        Image.fromarray(coords_img_data, mode="RGBA").save(output_folder / f"{i}.coords.png")
        Image.fromarray(uv_img_data, mode="RGBA").save(output_folder / f"{i}.uv.png")
        end = time.time()
        print(f"Conversion time for frame {i}: {end-start}")