import json
from PIL import Image
import numpy as np
from pathlib import Path
from numba import cuda

# File used by earth for visualization
OUTPUT_FILE = Path.cwd().parent \
    .joinpath("public", "data", "weather", "current", "current-wind-surface-level-gfs-1.0.json")

def param_to_json(data, nx, ny, lo1, la1, lo2, la2):
    # Returns the JSON representation compatible with earth of the given data
    return {
            "header": {
                "nx": nx,
                "ny": ny,
                "lo1": lo1,
                "la1": la1,
                "lo2": lo2,
                "la2": la2,
                "dx": (lo2-lo1)/(nx-1),
                "dy": (la1-la2)/(ny-1),
                "parameterUnit": "m.s-1",
                "refTime": "2025-06-04T00:00:00Z",
                "forecastTime": 0,
            },
            "data": data
        }

def to_earth_json(lats, lons, u, v):
    # Converts coordinates and wind information to JSON for visualizing them on earth
    lo1 = float(np.min(lons))
    lo2 = float(np.max(lons))
    la1 = float(np.max(lats))
    la2 = float(np.min(lats))

    u_data = [float(val) for val in np.flipud(u).flatten()]
    v_data = [float(val) for val in np.flipud(v).flatten()]

    with open(OUTPUT_FILE, "w") as f:
        json.dump([
            param_to_json(u_data,
                u.shape[1], u.shape[0],
                lo1, la1, lo2, la2), 
            param_to_json(v_data,
                v.shape[1], v.shape[0],
                lo1, la1, lo2, la2)], 
            f)
        
def uint16_to_float(value, range):
    # Converts a 16-bit unsigned integer to a floating point value
    return float(value/65535.0*(range[1]-range[0])+range[0])

uint16_to_float_gpu = cuda.jit(device=True)(uint16_to_float)

def decode(rgba, range1, range2, cast=uint16_to_float):
    # Decodes RGBA channels and returns the original two values
    val1 = cast(int(rgba[0]) << 8 | int(rgba[1]), range1)
    val2 = cast(int(rgba[2]) << 8 | int(rgba[3]), range2)
    return val1, val2

decode_gpu = cuda.jit(device=True)(decode)

@cuda.jit
def extract_data_kernel(pixels, range1, range2, x, y):
    i, j = cuda.grid(2)
    if i < pixels.shape[0] and j < pixels.shape[1]:
        x[i, j], y[i, j] = decode_gpu(pixels[i, j], range1, range2, uint16_to_float_gpu)

def extract_data(pixels, range1, range2, x, y):
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            x[i, j], y[i, j] = decode(pixels[i, j], range1, range2)

def extract_data_from_png(png_file, file_type, use_cuda):
    # Extracts and returns data encoded into the given PNG file
    img = Image.open(png_file)
    pixels = np.array(img, dtype=np.uint8)

    x = np.zeros((img.height, img.width), dtype=np.float32)
    y = np.zeros((img.height, img.width), dtype=np.float32)

    if file_type == "coords":
        range1 = (-90, 90)
        range2 = (-180, 180)
    elif file_type == "wind":
        range1 = (-50, 50)
        range2 = (-50, 50)

    if use_cuda:
        pixels_device = cuda.to_device(pixels)
        x_device = cuda.to_device(x)
        y_device = cuda.to_device(y)

        threadsperblock = (16, 16)
        blockspergrid_x = int(np.ceil(pixels.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(pixels.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        extract_data_kernel[blockspergrid, threadsperblock](pixels_device, range1, range2, x_device, y_device)

        x_device.copy_to_host(x)
        y_device.copy_to_host(y)
    else:
        extract_data(pixels, range1, range2, x, y)
    return x, y

def png_to_earth_json(input_dir, time, use_cuda=True):
    # Visualize on earth the given file, by extracting information from the corresponding PNGs
    lats, lons = extract_data_from_png(input_dir.joinpath(f"{time}.coords.png"), "coords", use_cuda)
    u, v = extract_data_from_png(input_dir.joinpath(f"{time}.uv.png"), "wind", use_cuda)
    to_earth_json(lats, lons, u, v)