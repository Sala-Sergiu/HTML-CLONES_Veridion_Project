# group_visual.py
import os
import argparse
import time
import json
import csv
import logging
from pathlib import Path
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx

# Importă funcțiile din visual_similarity.py
from visual_similarity import (
    get_visual_features, 
    compare_screenshots, 
    start_http_server
)

# Configurează loggerul global
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="output/execution.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def compute_features_for_all(input_dir, screenshot_dir, server_port, project_root, max_workers=1):
    """
    Parcurge toate fișierele HTML din input_dir și calculează pentru fiecare:
      - features (proprietăți CSS extrase – chiar dacă nu le vom folosi în grupare)
      - screenshot_path (calea screenshot-ului)
    Returnează:
      - features_dict: {filepath: (features, screenshot_path)}
      - errors: listă cu erori pentru fișierele care nu au putut fi procesate.
    """
    features_dict = {}
    errors = []
    files = list(Path(input_dir).glob("*.html"))
    
    def process_file(f):
        try:
            logger.info(f"Procesăm fișierul: {f.name}")
            feats, ss_path = get_visual_features(
                f, screenshot_dir=screenshot_dir, server_port=server_port, project_root=project_root
            )
            if feats is not None:
                logger.info(f"Fișierul {f.name} procesat cu succes.")
                return (f, feats, ss_path)
            else:
                error_msg = "Pagina este invalidă sau nu a putut fi procesată"
                logger.info(f"Fișierul {f.name} a fost ignorat: {error_msg}.")
                errors.append({"file": f.name, "error": error_msg})
                return None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Eroare la procesarea fișierului {f.name}: {error_msg}")
            errors.append({"file": f.name, "error": error_msg})
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, f): f for f in files}
        for future in as_completed(future_to_file):
            result = future.result()
            if result is not None:
                f, feats, ss_path = result
                features_dict[f] = (feats, ss_path)
    return features_dict, errors

def build_similarity_graph(features_dict, image_threshold, max_workers=1):
    """
    Construiște un graf în care fiecare nod este un fișier HTML.
    Se compară doar pe baza similarității imaginii (SSIM).
    Pentru fiecare pereche de fișiere, se calculează scorul SSIM și, dacă acesta depășește image_threshold,
    se adaugă o muchie între noduri.
    Returnează graful construit.
    """
    G = nx.Graph()
    files = list(features_dict.keys())
    G.add_nodes_from(files)
    
    edges_to_add = []
    pairs = list(combinations(files, 2))

    def process_pair(pair):
        f1, f2 = pair
        _, ss_path1 = features_dict[f1]
        _, ss_path2 = features_dict[f2]
        image_score = compare_screenshots(ss_path1, ss_path2)
        if image_score >= image_threshold:
            return (f1, f2, image_score)
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(process_pair, pairs):
            if result is not None:
                f1, f2, image_score = result
                logger.info(f"Adăugăm muchie între {f1.name} și {f2.name}: SSIM={image_score:.2f}")
                edges_to_add.append((f1, f2))
    
    G.add_edges_from(edges_to_add)
    return G

def save_groups(groups, json_file, csv_file):
    """
    Salvează grupurile într-un fișier JSON și CSV.
    Fiecare grup este o listă de nume de fișiere.
    """
    groups_list = []
    for idx, group in enumerate(groups, start=1):
        group_files = [f.name for f in group]
        groups_list.append({"group_id": idx, "files": group_files})
    
    with open(json_file, "w", encoding="utf-8") as jf:
        json.dump(groups_list, jf, indent=2)
    
    with open(csv_file, "w", encoding="utf-8", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["group_id", "file_name"])
        for group in groups_list:
            gid = group["group_id"]
            for file_name in group["files"]:
                writer.writerow([gid, file_name])
    
    logger.info(f"Grupurile au fost salvate în {json_file} și {csv_file}.")

def save_errors(errors, error_file):
    """
    Salvează lista de erori într-un fișier JSON.
    Fiecare eroare conține numele fișierului și un mesaj descriptiv.
    """
    with open(error_file, "w", encoding="utf-8") as ef:
        json.dump(errors, ef, indent=2)
    logger.info(f"Erorile au fost salvate în {error_file}.")

def organize_screenshots(groups, screenshot_dir="screenshots", tier_name="tier_1"):
    """
    Creează subfoldere (group_1, group_2, etc.) în directorul screenshots/<tier_name>/
    și mută fișierele PNG aferente fiecărui fișier HTML din screenshot_dir în acel subfolder.
    
    Exemplu final:
      screenshots/
        └── tier3/
            ├── group_1/
            ├── group_2/
            ...
    """
    tier_dir = Path(screenshot_dir) / tier_name
    tier_dir.mkdir(parents=True, exist_ok=True)
    for i, group in enumerate(groups, start=1):
        group_dir = tier_dir / f"group_{i}"
        group_dir.mkdir(exist_ok=True)
        for html_file in group:
            png_name = f"{html_file.stem}.png"
            src = Path(screenshot_dir) / png_name
            dest = group_dir / png_name
            if src.exists():
                try:
                    src.rename(dest)
                    logger.info(f"Mutat {png_name} în {group_dir}")
                except Exception as e:
                    logger.error(f"Eroare la mutarea {png_name}: {e}")
            else:
                logger.warning(f"Fișierul {png_name} nu există în {screenshot_dir}!")

# -----------------------
# Codul principal: CLI și orchestrare
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grupează fișierele HTML dintr-un director pe baza similarității imaginii (SSIM)."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directorul cu fișiere HTML (ex: clones/tier3)")
    parser.add_argument("--screenshot_dir", type=str, default="screenshots", help="Directorul pentru screenshot-uri")
    parser.add_argument("--output_json", type=str, default="output/tier3.json", help="Fișierul JSON de output")
    parser.add_argument("--output_csv", type=str, default="output/tier3.csv", help="Fișierul CSV de output")
    parser.add_argument("--error_output", type=str, default="output/errors.json", help="Fișierul JSON pentru erori")
    parser.add_argument("--image_threshold", type=float, default=0.9, help="Pragul de similaritate SSIM (default: 0.9)")
    parser.add_argument("--port", type=int, default=8000, help="Portul pentru serverul HTTP local")
    parser.add_argument("--max_workers_features", type=int, default=1, help="Numărul maxim de fire pentru procesarea fișierelor")
    parser.add_argument("--max_workers_pairs", type=int, default=1, help="Numărul maxim de fire pentru compararea perechilor")
    args = parser.parse_args()

    # Creăm directoarele pentru output, dacă nu există
    os.makedirs(Path(args.output_json).parent, exist_ok=True)
    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    os.makedirs(Path(args.error_output).parent, exist_ok=True)

    project_root = Path(os.getcwd())
    httpd = start_http_server(project_root, port=args.port)
    time.sleep(2)  # Așteptăm să se inițializeze serverul

    features_dict, errors = compute_features_for_all(
        args.input_dir, args.screenshot_dir, args.port, project_root, max_workers=args.max_workers_features
    )
    if not features_dict:
        logger.error("Nu s-au găsit fișiere HTML valide. Ieșim.")
        if errors:
            save_errors(errors, args.error_output)
        httpd.shutdown()
        exit()

    G = build_similarity_graph(
        features_dict, args.image_threshold, max_workers=args.max_workers_pairs
    )
    
    groups = list(nx.connected_components(G))
    logger.info(f"Numărul de grupuri găsite: {len(groups)}")
    
    save_groups(groups, args.output_json, args.output_csv)
    if errors:
        save_errors(errors, args.error_output)
    
    tier_name = Path(args.input_dir).name
    organize_screenshots(groups, args.screenshot_dir, tier_name)
    
    httpd.shutdown()
    logger.info("Procesarea s-a încheiat. Serverul a fost oprit.")
