import os
import argparse
from glob import glob
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to input data")
    parser.add_argument("--output_data", type=str, help="Path to output data")
    args = parser.parse_args()

    input_dir = args.data
    output_dir = args.output_data
    size = (28, 28)  # Resize target

    os.makedirs(output_dir, exist_ok=True)

    print("Converting PNG to JPG and resizing...")
    for file in glob(os.path.join(input_dir, "*.png")):
        try:
            img = Image.open(file).convert("RGB")  # Ensure no alpha channel
            img_resized = img.resize(size)

            filename = os.path.splitext(os.path.basename(file))[0] + ".jpg"
            output_file = os.path.join(output_dir, filename)

            img_resized.save(output_file, "JPEG")
            print(f"Saved: {output_file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()
