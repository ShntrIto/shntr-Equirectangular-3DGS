import os
import numpy as np
from PIL import Image
import argparse

def convert_rgb_to_rgba(rgb_dir, mask_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png') or f.endswith('.jpg')]

    for rgb_file in rgb_files:
        rgb_path = os.path.join(rgb_dir, rgb_file)
        mask_path = os.path.join(mask_dir, rgb_file)
        
        if not os.path.exists(mask_path):
            print(f"Mask for {rgb_file} not found, skipping.")
            continue

        rgb_image = Image.open(rgb_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")
        mask_image = mask_image.point(lambda p: 255 if p > 0 else 0)

        rgba_image = Image.merge("RGBA", (*rgb_image.split(), mask_image))
        # import pdb; pdb.set_trace()

        output_path = os.path.join(output_dir, rgb_file)
        rgba_image.save(output_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert RGB images to RGBA using mask images.")
    parser.add_argument("rgb_dir", type=str, help="Directory containing RGB images")
    parser.add_argument("mask_dir", type=str, help="Directory containing mask images")
    parser.add_argument("output_dir", type=str, help="Directory to save RGBA images")

    args = parser.parse_args()

    convert_rgb_to_rgba(args.rgb_dir, args.mask_dir, args.output_dir)