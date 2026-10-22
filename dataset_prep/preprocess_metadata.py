import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_tag_text(root, tag_name):
    """
    Finds the text content of a tag anywhere in the XML tree, 
    handling namespaces robustly.
    """
    # Direct XPath search
    elem = root.find(f".//{tag_name}")
    if elem is not None and elem.text is not None:
        return elem.text.strip()
    
    # Fallback to local name match to handle namespaces
    for el in root.iter():
        local_tag = el.tag.split('}')[-1] if '}' in el.tag else el.tag
        if local_tag == tag_name and el.text is not None:
            return el.text.strip()
            
    return None

def parse_metadata_file(file_path):
    """
    Parses a single Landsat 9 XML metadata file.
    Returns a tuple (product_id, metadata_dict) or None if parsing fails.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Error parsing XML file {file_path}: {e}")
        return None

    # Extract raw values
    product_id = find_tag_text(root, "LANDSAT_PRODUCT_ID")
    rad_mult_str = find_tag_text(root, "RADIANCE_MULT_BAND_10")
    rad_add_str = find_tag_text(root, "RADIANCE_ADD_BAND_10")
    k1_str = find_tag_text(root, "K1_CONSTANT_BAND_10")
    k2_str = find_tag_text(root, "K2_CONSTANT_BAND_10")
    date_acquired_str = find_tag_text(root, "DATE_ACQUIRED")
    scene_time_str = find_tag_text(root, "SCENE_CENTER_TIME")
    sun_azimuth_str = find_tag_text(root, "SUN_AZIMUTH")
    sun_elevation_str = find_tag_text(root, "SUN_ELEVATION")
    
    # Corners for center coordinate calculation
    ul_lat = find_tag_text(root, "CORNER_UL_LAT_PRODUCT")
    ul_lon = find_tag_text(root, "CORNER_UL_LON_PRODUCT")
    ur_lat = find_tag_text(root, "CORNER_UR_LAT_PRODUCT")
    ur_lon = find_tag_text(root, "CORNER_UR_LON_PRODUCT")
    ll_lat = find_tag_text(root, "CORNER_LL_LAT_PRODUCT")
    ll_lon = find_tag_text(root, "CORNER_LL_LON_PRODUCT")
    lr_lat = find_tag_text(root, "CORNER_LR_LAT_PRODUCT")
    lr_lon = find_tag_text(root, "CORNER_LR_LON_PRODUCT")

    # Verify presence of essential tags
    missing = []
    if not product_id: missing.append("LANDSAT_PRODUCT_ID")
    if not rad_mult_str: missing.append("RADIANCE_MULT_BAND_10")
    if not rad_add_str: missing.append("RADIANCE_ADD_BAND_10")
    if not k1_str: missing.append("K1_CONSTANT_BAND_10")
    if not k2_str: missing.append("K2_CONSTANT_BAND_10")
    if not date_acquired_str: missing.append("DATE_ACQUIRED")
    if not scene_time_str: missing.append("SCENE_CENTER_TIME")
    if not sun_azimuth_str: missing.append("SUN_AZIMUTH")
    if not sun_elevation_str: missing.append("SUN_ELEVATION")
    if not all([ul_lat, ul_lon, ur_lat, ur_lon, ll_lat, ll_lon, lr_lat, lr_lon]): missing.append("CORNER_LAT_LON")

    if missing:
        logger.warning(f"File {file_path} is missing required tags: {', '.join(missing)}")
        return None

    try:
        # Convert values to expected types
        radiance_mult = float(rad_mult_str)
        radiance_add = float(rad_add_str)
        k1 = float(k1_str)
        k2 = float(k2_str)
        sun_azimuth = float(sun_azimuth_str)
        sun_elevation = float(sun_elevation_str)
        
        # Calculate Center Lat/Lon
        center_lat = (float(ul_lat) + float(ur_lat) + float(ll_lat) + float(lr_lat)) / 4.0
        center_lon = (float(ul_lon) + float(ur_lon) + float(ll_lon) + float(lr_lon)) / 4.0

        # Convert DATE_ACQUIRED to DOY
        dt = datetime.strptime(date_acquired_str, "%Y-%m-%d")
        doy = dt.timetuple().tm_yday

        # Convert SCENE_CENTER_TIME to fractional hours
        time_clean = scene_time_str.strip().rstrip('Z')
        time_parts = time_clean.split(':')
        if len(time_parts) < 3:
            raise ValueError(f"Unexpected time format: {scene_time_str}")
        h = float(time_parts[0])
        m = float(time_parts[1])
        s = float(time_parts[2])
        fractional_hour = h + m / 60.0 + s / 3600.0

        metadata = {
            "radiance_mult": radiance_mult,
            "radiance_add": radiance_add,
            "k1": k1,
            "k2": k2,
            "doy": doy,
            "fractional_hour": round(fractional_hour, 4),
            "sun_azimuth": round(sun_azimuth, 4),
            "sun_elevation": round(sun_elevation, 4),
            "center_lat": round(center_lat, 6),
            "center_lon": round(center_lon, 6)
        }
        return product_id, metadata

    except ValueError as e:
        logger.error(f"Data type conversion error in {file_path}: {e}")
        return None

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'data')
    output_json_path = os.path.join(output_dir, 'metadata_mapping.json')

    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        return

    logger.info(f"Scanning directory: {input_dir} for Landsat 9 metadata XML files...")

    metadata_mapping = {}
    processed_count = 0

    # Walk directory recursively
    for root_dir, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('_MTL.xml') or file.endswith('__MTL.xml'):
                full_path = os.path.join(root_dir, file)
                result = parse_metadata_file(full_path)
                if result:
                    prod_id, metadata = result
                    metadata_mapping[prod_id] = metadata
                    processed_count += 1
                    logger.info(f"Successfully processed: {prod_id}")

    logger.info(f"Processed {processed_count} metadata files successfully.")

    # Save to data/metadata_mapping.json
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_mapping, f, indent=4)
        logger.info(f"Saved aggregated metadata to: {output_json_path}")
    except Exception as e:
        logger.error(f"Failed to write output JSON to {output_json_path}: {e}")

if __name__ == '__main__':
    main()
