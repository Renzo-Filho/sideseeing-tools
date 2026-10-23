import argparse
import os
import csv
from pathlib import Path
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import matplotlib
import numpy as np
import torch
import sys

try:
    from tqdm import tqdm
except ImportError:
    print("Instale o tqdm para ver a barra de progresso: pip install tqdm")
    def tqdm(iterable, **kwargs): return iterable

def overlay_masks(image, masks):
    image = image.convert("RGBA")
    if masks.shape[0] == 0: return image
    
    masks_np = 255 * masks.cpu().numpy().astype(np.uint8)
    n_masks = masks_np.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
    colors = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for mask, color in zip(masks_np, colors):
        mask_img = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image

def main():
    parser = argparse.ArgumentParser(description="Processa 68k imagens com Batch Processing e Resume.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Diretório de entrada")
    parser.add_argument("--output", "-o", type=str, required=True, help="Diretório de saída")
    parser.add_argument("--prompt", "-p", type=str, default="sidewalk", help="Prompt único")
    parser.add_argument("--batch-size", "-b", type=int, default=8, help="Imagens por vez na GPU")
    parser.add_argument("--save-overlay", action="store_true")
    
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    text_prompt = args.prompt
    batch_size = args.batch_size

    if not input_path.exists():
        print(f"Erro: Entrada '{input_path}' não existe.")
        sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Configurar CSV (Modo Append)
    csv_file_path = output_path / "detections.csv"
    file_exists = csv_file_path.exists()
    
    csv_file = open(csv_file_path, mode='a', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    
    if not file_exists:
        writer.writerow(["image_name", "relative_path", "prompt", "num_detections", "scores"])

    # 2. Listar arquivos
    print("Mapeando arquivos...")
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    all_files = []
    
    for ext in extensions:
        for file_path in input_path.rglob(ext):
            # Filtro 1: Ignorar se estiver dentro de uma pasta 'thumbs'
            if "thumbs" in file_path.parts:
                continue
            
            # Filtro 2: Ignorar arquivos ocultos (que começam com ponto)
            if file_path.name.startswith("."):
                continue
                
            all_files.append(file_path)
    
    print(f"Total de imagens encontradas (filtradas): {len(all_files)}")

    # 3. Filtrar imagens já processadas (Resume)
    files_to_process = []
    for img_file in all_files:
        rel_path = img_file.relative_to(input_path)
        dest_folder = output_path / rel_path.parent
        base_name = img_file.stem
        expected_mask = dest_folder / f"{base_name}_mask.png"
        
        if not expected_mask.exists():
            files_to_process.append(img_file)
    
    print(f"Imagens restantes para processar: {len(files_to_process)}")
    if len(files_to_process) == 0:
        print("Tudo já foi processado!")
        csv_file.close()
        return

    # 4. Carregar Modelo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM3 on {device}...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    # 5. Loop em Batches
    for i in tqdm(range(0, len(files_to_process), batch_size), desc="Processing Batches"):
        batch_files = files_to_process[i : i + batch_size]
        
        batch_images_pil = []
        valid_batch_files = []

        for img_file in batch_files:
            try:
                img = Image.open(img_file).convert("RGB")
                batch_images_pil.append(img)
                valid_batch_files.append(img_file)
            except Exception as e:
                print(f"Erro ao abrir {img_file}: {e}")
                writer.writerow([img_file.name, "ERROR", text_prompt, -1, str(e)])
        
        if not batch_images_pil:
            continue

        try:
            input_prompts = [text_prompt] * len(batch_images_pil)
            
            inputs = processor(
                images=batch_images_pil, 
                text=input_prompts, 
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            results_list = processor.post_process_instance_segmentation(
                outputs, 
                threshold=0.5, 
                mask_threshold=0.5, 
                target_sizes=[img.size[::-1] for img in batch_images_pil]
            )

            # 6. Salvar resultados
            for img_file, image_pil, result in zip(valid_batch_files, batch_images_pil, results_list):
                masks = result["masks"]
                scores = result["scores"].cpu().numpy().tolist()

                rel_path = img_file.relative_to(input_path)
                dest_folder = output_path / rel_path.parent
                dest_folder.mkdir(parents=True, exist_ok=True)
                base_name = img_file.stem

                if args.save_overlay:
                    # Overlay
                    viz_image = overlay_masks(image_pil.copy(), masks)
                    viz_image.save(dest_folder / f"{base_name}_overlay.png")

                # Mask & CSV
                if masks.shape[0] > 0:
                    combined_mask = torch.any(masks, dim=0).cpu().numpy().astype(np.uint8) * 255
                    Image.fromarray(combined_mask).save(dest_folder / f"{base_name}_mask.png")
                    scores_str = ";".join([f"{s:.4f}" for s in scores])
                else:
                    empty_mask = Image.new("L", image_pil.size, 0)
                    empty_mask.save(dest_folder / f"{base_name}_mask.png") 
                    scores_str = ""

                writer.writerow([img_file.name, str(rel_path), text_prompt, len(masks), scores_str])
            
            csv_file.flush()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"| ERRO: GPU sem memória. Reduza o --batch-size (Atual: {batch_size})")
                sys.exit(1)
            else:
                print(f"Erro no batch {i}: {e}")

    csv_file.close()
    print("Processamento concluído.")

if __name__ == "__main__":
    main()
