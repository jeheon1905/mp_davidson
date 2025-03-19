import argparse
from PIL import Image
import os
import sys


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Horizontally merge multiple PNG images."
    )
    parser.add_argument(
        "--plot",
        type=str,
        required=True,
        choices=["eigval", "residual"],
        help="Plot type: eigval or residual",
    )
    parser.add_argument(
        "--supercell", type=str, required=True, help="Supercell size, e.g., 1_1_5"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        required=False,
        default=[
            "DP",
            "DP_SP4precond",
            "MP_scheme1",
            "MP_scheme2",
            "MP_scheme3",
            "MP_scheme4",
            "MP_scheme5",
            "SP",
        ],
        help="List of methods to process. Default includes multiple schemes.",
    )
    parser.add_argument(
        "--recalc_convg_history",
        action="store_true",
        help="Recalculate convergence history",
    )
    return parser.parse_args()


def load_images(image_paths):
    """Load images from file paths and handle errors if any are missing."""
    images = []

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"[Warning] Image not found: {img_path}", file=sys.stderr)
            continue

        try:
            img = Image.open(img_path)
            images.append(img)
        except Exception as e:
            print(f"[Error] Failed to open image: {img_path}\n{e}", file=sys.stderr)

    if not images:
        print("[Error] No images loaded. Exiting...", file=sys.stderr)
        sys.exit(1)

    return images


def merge_images_horizontally(images):
    """Merge a list of images horizontally into a single image."""
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)
    max_height = max(heights)

    # Create a new blank image (RGBA supports transparency)
    merged_image = Image.new("RGBA", (total_width, max_height))

    # Paste each image side by side
    x_offset = 0
    for img in images:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    return merged_image


def main():
    args = parse_arguments()

    plot = args.plot
    supercell = args.supercell
    methods = args.methods

    print(f"[Info] Plot type   : {plot}")
    print(f"[Info] Supercell   : {supercell}")
    print(f"[Info] Methods     : {methods}")

    # Create the list of image file paths
    image_files = [
        (
            f"./expt.CNT_6_0/History.{plot}.{supercell}_{method}.png"
            if not args.recalc_convg_history
            else f"./expt.CNT_6_0/History.{plot}.{supercell}_{method}.recalc_convg_history.png"
        )
        for method in methods
    ]
    print(f"Debug: image_files = {image_files}")

    print(f"[Info] Loading {len(image_files)} images...")

    # Load images from files
    images = load_images(image_files)

    print(f"[Info] Successfully loaded {len(images)} images. Merging...")

    # Merge the loaded images horizontally
    merged_image = merge_images_horizontally(images)

    # Define output file name
    output_filename = (
        f"merged_{plot}_{supercell}.png"
        if not args.recalc_convg_history
        else f"merged_{plot}_{supercell}_recalc_convg_history.png"
    )

    # Save merged image
    merged_image.save(output_filename)

    print(f"[Success] Merged image saved as: {output_filename}")


if __name__ == "__main__":
    main()
