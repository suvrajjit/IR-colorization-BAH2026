import os
import zipfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def zip_essential_data(source_dir: str, zip_filepath: str):
    """
    Scans the source patches directory recursively and packages ONLY the essential .npy files,
    excluding large visual verification .png files and intermediate folders (like downscaled_data
    and rgb_images) from the final zip archive.
    """
    logger.info(f"Scanning directory: {source_dir} for essential .npy files...")
    
    file_count = 0
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.npy'):
                    abs_path = os.path.join(root, file)
                    # We want the path inside the zip file to start with "output/patches/..."
                    # Find the relative path from the project root (parent directory of output)
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(source_dir)))
                    rel_path = os.path.relpath(abs_path, project_root)
                    
                    zip_file.write(abs_path, arcname=rel_path)
                    file_count += 1
                    if file_count % 100 == 0:
                        logger.info(f"Added {file_count} files to zip...")
                        
    logger.info(f"Successfully zipped {file_count} essential .npy files into: {zip_filepath}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_patches_dir = os.path.join(base_dir, "output", "patches")
    target_zip = os.path.join(base_dir, "output_.zip")
    
    zip_essential_data(source_patches_dir, target_zip)
