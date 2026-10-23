import argparse
import pandas as pd
import imageio
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def extract_parts(img_path):
    base = os.path.basename(img_path).replace(".jpg", "")
    prefix, idx, ms = base.rsplit("_", 2)
    return prefix, idx, ms


def save_gif(args):
    group, col, col_dir, fps = args
    imgs = [imageio.imread(p) for p in group if os.path.exists(p)]
    if not imgs:
        return None

    prefix, fidx, ms = extract_parts(group[0])
    _, lidx, _ = extract_parts(group[-1])

    gif_name = f"{prefix}_{fidx}_{lidx}_{ms}.gif"
    out_path = os.path.join(col_dir, gif_name)
    imageio.mimsave(out_path, imgs, fps=fps)

    return [col, os.path.basename(group[0]), os.path.basename(group[-1]), gif_name]


def generate_groups(df, col, img_base):
    groups = []
    group = []
    zeroes = 0

    for i, val in enumerate(df[col]):
        img_path = os.path.join(img_base, df.loc[i, "image"])

        if val == 1:
            group.append(img_path)
            zeroes = 0
        else:
            zeroes += 1
            if zeroes <= 3:  # tolerância
                continue
            if group:
                groups.append(group)
            group = []
            zeroes = 0

    if group:
        groups.append(group)

    return groups


def process(input_csv, output_dir, img_base, fps):
    df = pd.read_csv(input_csv)
    cols = ["crosswalk", "curbramp", "surfaceproblem", "obstacle"]
    summary = []

    os.makedirs(output_dir, exist_ok=True)

    tasks = []

    print("\n🔎 Detectando grupos por coluna...\n")

    for col in cols:
        col_dir = os.path.join(output_dir, col)
        os.makedirs(col_dir, exist_ok=True)

        groups = generate_groups(df, col, img_base)

        print(f"• {col}: {len(groups)} grupos detectados")

        for g in groups:
            tasks.append((g, col, col_dir, fps))

    print("\n🎞️ Gerando GIFs...\n")

    with Pool(cpu_count()) as p:
        for result in tqdm(p.imap_unordered(save_gif, tasks), total=len(tasks)):
            if result:
                summary.append(result)

    summary_path = os.path.join(output_dir, "summary.csv")
    pd.DataFrame(summary, columns=["column", "start_img", "end_img", "gif_name"]).to_csv(summary_path, index=False)

    print("\n📄 summary.csv salvo em:", summary_path)
    print("✔ Finalizado\n")


def main():
    ap = argparse.ArgumentParser(description="Generate GIFs from detection CSV with tolerance, multiprocessing and logs.")
    ap.add_argument("--csv", required=True, help="Input CSV")
    ap.add_argument("--images", required=True, help="Base directory of images")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--fps", type=int, default=2, help="GIF FPS")
    args = ap.parse_args()

    process(args.csv, args.out, args.images, args.fps)


if __name__ == "__main__":
    main()

