import os
import sys
import argparse
import requests
import logging
from pystac_client import Client
import planetary_computer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url, dest_path, retries=5, timeout=20):
    """Downloads a file from a URL with resume support, timeouts, and automatic retries."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_dest = dest_path + ".tmp"
    
    for attempt in range(retries):
        try:
            headers = {}
            downloaded = 0
            if os.path.exists(temp_dest):
                downloaded = os.path.getsize(temp_dest)
                headers['Range'] = f'bytes={downloaded}-'
                
            response = requests.get(url, stream=True, headers=headers, timeout=timeout)
            
            if response.status_code == 206:
                # Resuming successfully
                mode = 'ab'
            elif response.status_code == 200:
                # Started from scratch
                mode = 'wb'
                downloaded = 0
            else:
                response.raise_for_status()
                
            total_size = int(response.headers.get('content-length', 0)) + downloaded
            
            chunk_size = 512 * 1024  # 512 KB chunks
            with open(temp_dest, mode) as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            sys.stdout.write(f"\rDownloading: {percent:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)")
                        else:
                            sys.stdout.write(f"\rDownloading: {downloaded / (1024*1024):.1f} MB")
                        sys.stdout.flush()
            print()
            os.replace(temp_dest, dest_path)
            return  # Success
        except (requests.RequestException, IOError) as e:
            logger.warning(f"\nConnection error: {e}. Retrying ({attempt + 1}/{retries})...")
            import time
            time.sleep(2)
            
    # If all retries fail
    if os.path.exists(temp_dest):
        os.remove(temp_dest)
    raise Exception(f"Failed to download after {retries} attempts.")

def main():
    parser = argparse.ArgumentParser(description="Download Landsat 9 full scenes over India from Microsoft Planetary Computer.")
    parser.add_argument("--num_images", type=int, default=15, help="Number of unique satellite images to download.")
    parser.add_argument("--start_date", type=str, default="2025-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default="2025-12-31", help="End date (YYYY-MM-DD).")
    parser.add_argument("--output_dir", type=str, default="./input", help="Directory to save downloaded products.")
    args = parser.parse_args()

    # Bounding box of India [min_lon, min_lat, max_lon, max_lat]
    india_bbox = [68.0, 8.0, 97.0, 37.0]

    logger.info("Connecting to Microsoft Planetary Computer STAC Catalog...")
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

    logger.info(f"Searching Landsat 9 scenes over India from {args.start_date} to {args.end_date} with < 20% cloud cover...")
    search = catalog.search(
        collections=["landsat-c2-l2"],
        bbox=india_bbox,
        datetime=f"{args.start_date}/{args.end_date}",
        query={
            "eo:cloud_cover": {"lt": 20},
            "platform": {"eq": "landsat-9"}
        },
        max_items=500  # Pull a large enough catalog page to filter
    )

    items = list(search.item_collection())
    logger.info(f"Found {len(items)} matching scenes.")

    if not items:
        logger.error("No scenes found matching the criteria.")
        return

    # Group items by WRS Path and Row to ensure geographic diversity
    unique_scenes = {}
    for item in items:
        wrs_path = item.properties.get("landsat:wrs_path")
        wrs_row = item.properties.get("landsat:wrs_row")
        key = (wrs_path, wrs_row)
        
        # Keep the scene with the lowest cloud cover for this path/row
        if key not in unique_scenes or item.properties.get("eo:cloud_cover") < unique_scenes[key].properties.get("eo:cloud_cover"):
            unique_scenes[key] = item

    selected_items = list(unique_scenes.values())
    # Sort by cloud cover and take the requested number
    selected_items.sort(key=lambda x: x.properties.get("eo:cloud_cover"))
    selected_items = selected_items[:args.num_images]

    logger.info(f"Selected {len(selected_items)} unique geographic scenes for download:")
    for idx, item in enumerate(selected_items):
        path = item.properties.get("landsat:wrs_path")
        row = item.properties.get("landsat:wrs_row")
        cc = item.properties.get("eo:cloud_cover")
        logger.info(f"  [{idx+1}] ID: {item.id} | Path/Row: {path}/{row} | Cloud Cover: {cc}%")

    # Define asset key mapping to target files
    asset_mapping = {
        "blue": "B2.TIF",
        "green": "B3.TIF",
        "red": "B4.TIF",
        "lwir11": "B10.TIF",
        "mtl.xml": "MTL.xml"
    }

    for idx, item in enumerate(selected_items):
        logger.info(f"--- Downloading Scene [{idx+1}/{len(selected_items)}]: {item.id} ---")
        product_dir = os.path.join(args.output_dir, f"landsat9_{item.id}_product")
        os.makedirs(product_dir, exist_ok=True)

        signed_item = planetary_computer.sign(item)

        for asset_key, file_suffix in asset_mapping.items():
            filename = f"landsat9_{item.id}_{file_suffix}"
            dest_path = os.path.join(product_dir, filename)

            if os.path.exists(dest_path):
                logger.info(f"  Asset {asset_key} ({file_suffix}) already exists, skipping download.")
                continue

            asset = signed_item.assets.get(asset_key)
            if not asset:
                logger.error(f"  Asset {asset_key} not found in STAC metadata.")
                continue

            url = asset.href
            logger.info(f"  Downloading Asset {asset_key} ({file_suffix})...")
            try:
                download_file(url, dest_path)
            except Exception as e:
                logger.error(f"  Failed to download asset {asset_key} ({file_suffix}): {e}")
                # Clean up if download failed
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                return

    logger.info("All selected scenes successfully downloaded!")

if __name__ == "__main__":
    main()
