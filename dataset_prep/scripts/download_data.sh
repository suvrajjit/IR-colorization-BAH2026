#!/bin/bash
# Wrapper for the MPC download script
echo "Starting data download via Microsoft Planetary Computer..."
.venv/Scripts/python scripts/download_mpc.py --num_images 15 --start_date "2025-01-01" --end_date "2025-12-31" --output_dir "./input"
