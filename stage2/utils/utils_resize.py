import argparse
import sys
from pathlib import Path

from PIL import Image


def crop_left_square(input_image_path: str, n: int, output_image_path: str) -> None:
    """
    Crop an image to an n x n square anchored at the top-left corner.

    If n is larger than the shorter side of the image, the shorter side is used.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    input_path = Path(input_image_path)
    output_path = Path(output_image_path)

    with Image.open(input_path) as image:
        width, height = image.size
        crop_size = min(n, width, height)
        cropped = image.crop((0, 0, crop_size, crop_size))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop image to a top-left n x n square. If n is larger than the shorter side, use the shorter side."
    )
    parser.add_argument("--input-image", required=True, help="Input image path")
    parser.add_argument("--n", required=True, type=int, help="Target square size")
    parser.add_argument("--output-image", required=True, help="Output image path")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(2)

    args = parser.parse_args()
    crop_left_square(args.input_image, args.n, args.output_image)


if __name__ == "__main__":
    main()
