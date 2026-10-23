from pathlib import Path

import argparse
import os
import subprocess


def process_images(input_dir, output_dir, width, height, quality):
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    converted = 0
    skipped = 0
    ignored = 0
    errors = 0

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.startswith("."):
                ignored += 1
                continue

            if file.lower().endswith(exts):
                input_path = Path(root) / file
                relative_path = input_path.relative_to(input_dir)
                output_path = Path(output_dir) / relative_path.with_suffix(".jpg")

                if output_path.exists():
                    skipped += 1
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)

                cmd = ["ffmpeg", "-y", "-i", str(input_path)]

                if width or height:
                    w, h = width or -1, height or -1
                    cmd += ["-vf", f"scale={w}:{h}"]

                cmd += ["-q:v", str(quality), str(output_path)]

                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    converted += 1
                except Exception:
                    errors += 1

    print("\n--- RESUMO ---")
    print(f"Convertidas: {converted}")
    print(f"Puladas:     {skipped}")
    print(f"Ignoradas:   {ignored}")
    print(f"Erros:       {errors}")

def main():
    parser = argparse.ArgumentParser(description="Reduz e converte imagens para JPEG mantendo estrutura.")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--quality", type=int, default=12)
    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir, args.width, args.height, args.quality)

if __name__ == "__main__":
    main()
