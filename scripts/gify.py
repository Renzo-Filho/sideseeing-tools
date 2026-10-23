import pandas as pd
import imageio
import os
import argparse

def extract_name(img_path):
    """Extrai prefixo e índices da imagem"""
    base = os.path.basename(img_path).replace(".jpg", "")
    parts = base.rsplit("_", 2)  # ex: ['2025-11-09-07-23-58-838', '00092', 'ms']
    prefix, index, ms = parts
    return prefix, index, ms

def create_gifs_with_csv(input_csv, output_dir, img_base_dir, fps, summary_csv):
    df = pd.read_csv(input_csv)
    columns = ["crosswalk", "curbramp", "surfaceproblem", "obstacle"]

    os.makedirs(output_dir, exist_ok=True)
    summary = []

    for col in columns:
        col_dir = os.path.join(output_dir, col)
        os.makedirs(col_dir, exist_ok=True)

        mask = df[col] == 1
        group = []

        for i, val in enumerate(mask):
            if val:
                img_path = os.path.join(img_base_dir, df.loc[i, "image"])
                group.append(img_path)
            if not val or i == len(mask) - 1:
                if group:
                    images = [imageio.imread(img) for img in group if os.path.exists(img)]
                    if images:
                        prefix, first_index, ms = extract_name(group[0])
                        _, last_index, _ = extract_name(group[-1])
                        if first_index == last_index:
                            gif_name = f"{prefix}_{first_index}_{ms}.gif"
                        else:
                            gif_name = f"{prefix}_{first_index}_{last_index}_{ms}.gif"
                        gif_path = os.path.join(col_dir, gif_name)
                        imageio.mimsave(gif_path, images, fps=fps)

                        # Adicionar ao CSV resumo
                        summary.append({
                            "column": col,
                            "gif_path": gif_path,
                            "first_image": group[0],
                            "last_image": group[-1]
                        })
                    group = []

        print(f"{col}: GIFs gerados em {col_dir}")

    # Salvar CSV resumo
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(summary_csv, index=False)
    print(f"CSV resumo salvo em: {summary_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerar GIFs de sequências consecutivas de 1 por coluna com CSV resumo.")
    parser.add_argument("input_csv", help="Caminho do CSV original")
    parser.add_argument("output_dir", help="Diretório base onde os subdiretórios serão criados")
    parser.add_argument("img_base_dir", help="Diretório base das imagens")
    parser.add_argument("summary_csv", help="Caminho do CSV resumo a ser gerado")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames por segundo do GIF (padrão 1.0)")
    args = parser.parse_args()

    create_gifs_with_csv(args.input_csv, args.output_dir, args.img_base_dir, args.fps, args.summary_csv)

