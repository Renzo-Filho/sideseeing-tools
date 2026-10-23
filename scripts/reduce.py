import pandas as pd
import argparse

def sample_group(series):
    """Retorna índices da primeira, do meio e da última ocorrência de 1 consecutiva"""
    indices = series.index.tolist()
    if not indices:
        return []
    if len(indices) <= 3:
        return indices
    mid = len(indices) // 2
    return [indices[0], indices[mid], indices[-1]]

def reduce_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    selected_rows = []

    for col in ["crosswalk", "curbramp", "surfaceproblem", "obstacle"]:
        mask = df[col] == 1
        group = []
        for i, val in enumerate(mask):
            if val:
                group.append(i)
            if not val or i == len(mask) - 1:
                if group:
                    selected_rows.extend(sample_group(df.loc[group, :]))
                    group = []

    # Remover duplicatas e ordenar
    selected_rows = sorted(set(selected_rows))
    df_sampled = df.loc[selected_rows]
    df_sampled.to_csv(output_csv, index=False)
    print(f"Arquivo reduzido salvo em: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduz CSV mantendo imagens representativas de sequências de eventos.")
    parser.add_argument("input_csv", help="Caminho do CSV original")
    parser.add_argument("output_csv", help="Caminho do CSV de saída")
    args = parser.parse_args()

    reduce_csv(args.input_csv, args.output_csv)

