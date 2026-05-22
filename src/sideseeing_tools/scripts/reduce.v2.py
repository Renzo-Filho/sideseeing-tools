import pandas as pd
import argparse

def reduce_csv_per_column(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    selected_rows = set()

    cols = ["crosswalk", "curbramp", "surfaceproblem", "obstacle"]

    for col in cols:
        mask = df[col] == 1
        group = []
        for i, val in enumerate(mask):
            if val:
                group.append(i)
            if not val or i == len(mask) - 1:
                if group:
                    # Mantém apenas a primeira imagem da sequência
                    selected_rows.add(group[0])
                    group = []

    # Ordenar os índices e salvar CSV reduzido
    selected_rows = sorted(selected_rows)
    df_sampled = df.loc[selected_rows]
    df_sampled.to_csv(output_csv, index=False)
    print(f"Arquivo reduzido salvo em: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduz CSV mantendo apenas a primeira imagem de cada sequência por coluna.")
    parser.add_argument("input_csv", help="Caminho do CSV original")
    parser.add_argument("output_csv", help="Caminho do CSV de saída")
    args = parser.parse_args()

    reduce_csv_per_column(args.input_csv, args.output_csv)

