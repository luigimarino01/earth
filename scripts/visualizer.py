import json
from PIL import Image
import numpy as np
from pathlib import Path
import struct

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
                "dx": (lo2-lo1)/nx,
                "dy": (la1-la2)/ny,
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

    u_data = [float(val) for val in u.flatten()]
    v_data = [float(val) for val in v.flatten()]

    with open(OUTPUT_FILE, "w") as f:
        json.dump([
            param_to_json(u_data,
                u.shape[1], u.shape[0],
                lo1, la1, lo2, la2), 
            param_to_json(v_data,
                v.shape[1], v.shape[0],
                lo1, la1, lo2, la2)], 
            f)
        
def uint16_to_float(value, min, max):
    # Converts a 16-bit unsigned integer to a floating point value
    return float(value/65535.0*(max-min)+min)

def decode(rgba, min1, max1, min2, max2):
    # Decodes RGBA channels and returns the original two values
    val1 = uint16_to_float(np.uint16(rgba[0]) << 8 | rgba[1], min1, max1)
    val2 = uint16_to_float(np.uint16(rgba[2]) << 8 | rgba[3], min2, max2)
    return val1, val2

def extract_data_from_png(png_file, file_type):
    # Extracts and returns data encoded into the given PNG file
    img = Image.open(png_file)
    pixels = img.load()

    x = np.zeros((img.height, img.width), dtype=np.float32)
    y = np.zeros((img.height, img.width), dtype=np.float32)
    for i in range(img.height):
        for j in range(img.width):
            if file_type == "coords":
                x[i, j], y[i, j] = decode(pixels[j, i], -90, 90, -180, 180)
            elif file_type == "wind":
                x[i, j], y[i, j] = decode(pixels[j, i], -50, 50, -50, 50)
    return x, y

def png_to_earth_json(input_dir, filename):
    # Visualize on earth the given file, by extracting information from the corresponding PNGs
    lats, lons = extract_data_from_png(input_dir.joinpath(filename + ".coords.png"), "coords")
    u, v = extract_data_from_png(input_dir.joinpath(filename + ".uv.png"), "wind")
    to_earth_json(lats, lons, u, v)

def deserialize(data, len, start, value_size=4):
    # Deserializes one or more floating point values
    end = start+len*value_size
    return np.array(struct.unpack("f"*len, data[start:end]), dtype=np.float32), end
    
def extract_data_from_bin(filepath):
    # Extracts coordinates and wind information from the given binary file
    with open(filepath, "rb") as f:
        data = f.read()

    # Gets the dimensions of U and V matrices
    ny = struct.unpack("I", data[0:4])[0]
    nx = struct.unpack("I", data[4:8])[0]

    # Extract geographical coordinates
    lats, offset = deserialize(data, ny, 8)
    lons, offset = deserialize(data, nx, offset)

    # Extract wind U and V components
    u, offset = deserialize(data, ny*nx, offset)
    v, offset = deserialize(data, ny*nx, offset)

    u = u.reshape((ny, nx))
    v = v.reshape((ny, nx))
    return lats, lons, u, v

def bin_to_earth_json(input_dir, filename):
    # Visualize on earth the given binary file
    lats, lons, u, v = extract_data_from_bin(input_dir.joinpath(filename + ".bin"))
    to_earth_json(lats, lons, u, v)