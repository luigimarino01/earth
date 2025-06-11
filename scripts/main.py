from pathlib import Path
import argparse

from converter import netcdf_to_bin, netcdf_to_png, load_data
from s3client import S3Client
from visualizer import bin_to_earth_json, png_to_earth_json, to_earth_json

NETCDF_DIR = Path.cwd().parent.joinpath("netcdf")
TMP_DIR = Path.cwd().parent.joinpath("tmp")

S3_BUCKET = "ccpaper001"
S3_DIR = Path("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="The name of the file in the netcdf directory to visualize.")
    parser.add_argument("--show-only", "-s", action="store_true", help="Only visualize the netCDF file on earth avoiding the conversion process.")
    parser.add_argument("--mode", "-m", choices=["bin", "png"], default="bin", help="Set the type of conversion to apply before uploading to S3.")
    args = parser.parse_args()

    if args.show_only:
        lats, lons, u, v = load_data(NETCDF_DIR, args.name)
        to_earth_json(lats, lons, u, v)
        exit(0)

    if args.mode == "bin":
        filename = args.name + ".bin"
        local_files = [str(TMP_DIR.joinpath(filename))]
        s3_files = [str(S3_DIR.joinpath(filename))]
    elif args.mode == "png":
        f1 = args.name + ".coords.png"
        f2 = args.name + ".uv.png"
        local_files = [str(TMP_DIR.joinpath(f1)), str(TMP_DIR.joinpath(f2))]
        s3_files = [str(S3_DIR.joinpath(f1)), str(S3_DIR.joinpath(f2))]

    client = S3Client()
    if client.exists(s3_files[0], S3_BUCKET):
        # Download the converted file(s) from S3 if they have been already uploaded
        print("Downloading converted files...")
        client.download(s3_files, S3_BUCKET, local_files)
    else:
        # Convert the netCDF file according to the selected mode
        print(f"Converting to {args.mode}...")
        if args.mode == "png":
            netcdf_to_png(NETCDF_DIR, args.name, TMP_DIR)
        else:
            netcdf_to_bin(NETCDF_DIR, args.name, TMP_DIR)

        # Upload the converted file(s) to S3
        print("Uploading to S3...")
        client.upload(local_files, S3_BUCKET, s3_files)

    print("Visualizing on earth...")
    if args.mode == "png":
        png_to_earth_json(TMP_DIR, args.name)
    elif args.mode == "bin":
        bin_to_earth_json(TMP_DIR, args.name)