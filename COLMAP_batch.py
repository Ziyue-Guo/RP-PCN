# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:35:22 2023

@author: AAA
"""

import subprocess
import os

# Define the list of folders
folders = [
    r"D:\rapeseed\GMJ_rapeseed\IMG_4759_frames",
    r"D:\rapeseed\GMJ_rapeseed\IMG_4761_frames",
    r"D:\rapeseed\GMJ_rapeseed\IMG_4764_frames",
    r"D:\rapeseed\GMJ_rapeseed\IMG_4767_frames",
    r"D:\rapeseed\GMJ_rapeseed\IMG_4769_frames",
    r"D:\rapeseed\GMJ_rapeseed\IMG_4771_frames",
    r"D:\rapeseed\GMJ_rapeseed\IMG_4773_frames",
    r"D:\rapeseed\GMJ_rapeseed\IMG_4775_frames"]

# Batch commands
batch_commands = """
@echo off
call conda activate instantNGP
echo Y | python D:/PhD_Study/Instant-NGP-for-RTX-3000-and-4000/Instant-NGP-for-RTX-3000-and-4000/scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 32
"""

# Iterate over folders and execute the command in each folder
for folder in folders:
    # Create a temporary batch file
    batch_file = os.path.join(folder, "temp_run.bat")
    with open(batch_file, "w") as bf:
        bf.write(batch_commands)

    # Execute the batch file
    subprocess.run(f"cmd /c {batch_file}", cwd=folder)

    # Delete the batch file
    os.remove(batch_file)
