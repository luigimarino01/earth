from pathlib import Path
import argparse
import numpy as np

from converter import netcdf_to_png, load_data
from s3client import S3Client
from visualizer import png_to_earth_json, to_earth_json

NETCDF_DIR = Path.cwd().parent.joinpath("netcdf")
TMP_DIR = Path.cwd().parent.joinpath("tmp")

S3_BUCKET = "ccpaper001"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="The name of the input NetCDF file to visualize. It must be located in the netcdf folder.")
    parser.add_argument("--cuda", "-c", action="store_true", help="If enabled it accelerates the encoding and deconding stages through CUDA, whenever its possible.")

    subparsers = parser.add_subparsers(dest="action")
    encode_parser = subparsers.add_parser("encode", help="Encode the given NetCDF file to PNGs.")

    view_parser = subparsers.add_parser("view", help="Decode PNG files associated to the given input and visualize them on Earth.")
    view_parser.add_argument("time", default=0, type=int, help="The specific time instant to visualize.")
    view_parser.add_argument("--use-netcdf", "-n", action="store_true", help="Take data from the base NetCDF file instead of decoding PNGs.")
    args = parser.parse_args()

    png_folder = TMP_DIR.joinpath(args.name)
    png_folder.mkdir(exist_ok=True, parents=True)

    client = S3Client()
    if args.action == "encode":
        # Encode netCDF file to PNG format
        print(f"Converting to PNG tiles...")
        netcdf_to_png(NETCDF_DIR, args.name, png_folder, args.cuda)

        # Upload PNG files to S3
        print("Uploading to S3...")
        #client.upload_folder(png_folder, S3_BUCKET)
    elif args.action == "view":
        if args.use_netcdf:
            lats, lons, u, v = load_data(NETCDF_DIR, args.name)
            u[args.time] = np.nan_to_num(u[args.time], nan=0)
            v[args.time] = np.nan_to_num(v[args.time], nan=0)
            to_earth_json(lats, lons, u[args.time], v[args.time])
        else:
            # Download the PNG files from S3 if they have been already uploaded
            print("Downloading converted files...")
            #client.download_folder(args.name, S3_BUCKET, TMP_DIR)

            # View data on Earth
            print("Visualizing on earth...")
            png_to_earth_json(png_folder, args.time, args.cuda)