import xarray as xr
import numpy as np
from PIL import Image
from pathlib import Path
import struct

VAR_U = "u"
VAR_V = "v"
VAR_LAT = "latitude"
VAR_LON = "longitude"

INPUT_DIR = Path("netcdf")
OUTPUT_DIR = Path("png")

def float_to_uint16(value, min, max):
    # Converts a 32-bit floating point value into a 16-bit unsigned integer
    return np.uint16(np.around((value-min)/(max-min)*65535))

def encode(val1, min1, max1, val2, min2, max2):
    # Encodes two 32-bit floating point values into RGBA channels
    z1 = float_to_uint16(val1, min1, max1)
    z2 = float_to_uint16(val2, min2, max2)
    return [z1 >> 8, z1 & 0xFF, z2 >> 8, z2 & 0xFF]

def load_data(input_dir, filename):
    # Loads the geographical coordinates and the U and V components from the given NetCDF file
    ds = xr.open_dataset(input_dir.joinpath(filename + ".nc.nc4"))
    lats = ds[VAR_LAT].values
    lons = ds[VAR_LON].values
    u = ds[VAR_U].squeeze().values
    v = ds[VAR_V].squeeze().values
    return lats, lons, u, v

def netcdf_to_png(input_dir, filename, output_dir):
    # Converts the given NetCDF file to two PNG files, by encoding data into RGBA channels
    lats, lons, u, v = load_data(input_dir, filename)

    output_folder = output_dir.joinpath(filename)
    output_folder.mkdir(exist_ok=True)
    for i in range(u.shape[0]):
        # Encode the extracted data
        coords_img_data = np.zeros((u.shape[1], u.shape[2], 4), dtype=np.uint8)
        uv_img_data = np.zeros((u.shape[1], u.shape[2], 4), dtype=np.uint8)
        for j in range(u.shape[1]):
            for k in range(u.shape[2]):
                coords_img_data[j, k] = encode(lats[j], -90, 90, lons[k], -180, 180)
                uv_img_data[j, k] = encode(u[i, j, k], -50, 50, v[i, j, k], -50, 50)

        # Store the output images
        coords_file = output_folder.joinpath(f"{i}.coords.png")
        Image.fromarray(coords_img_data, mode="RGBA").save(coords_file)
        uv_file = output_folder.joinpath(filename + f"{i}.uv.png")
        Image.fromarray(uv_img_data, mode="RGBA").save(uv_file)

def serialize(data):
    # Serialize one or more float values into their binary representations
    data = data.flatten()
    return struct.pack("f"*len(data), *data)

def netcdf_to_bin(input_dir, filename, output_dir):
    """ Extracts wind information from the given NetCDF file 
        and stores them into a binary file """
    lats, lons, u, v = load_data(input_dir, filename)
    with open(output_dir.joinpath(filename + ".bin"), "wb") as f:
        f.write(struct.pack("II", u.shape[0], u.shape[1]))
        f.write(serialize(lats))
        f.write(serialize(lons))
        f.write(serialize(u))
        f.write(serialize(v))