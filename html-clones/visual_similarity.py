# visual_similarity.py
import os
import threading
import socketserver
import http.server
import argparse
from pathlib import Path
import time
import logging

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from playwright.sync_api import sync_playwright, TimeoutError

# Configurează loggerul
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

# -----------------------
# Server HTTP local
# -----------------------
def start_http_server(directory: Path, port: int = 8000, timeout: int = 10):
    """
    Pornește un server HTTP local în directorul specificat pe portul dat.
    Returnează instanța serverului.
    """
    os.chdir(directory)  # Schimbă directorul de lucru la directorul rădăcină
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)

    def serve():
        with httpd:
            httpd.timeout = timeout
            httpd.serve_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    logger.info(f"HTTP server started on http://localhost:{port} serving directory {directory}")
    return httpd

# -----------------------
# Auto-scroll pentru pagini (adaptiv)
# -----------------------
def auto_scroll(page, pause_time=0.3):
    """
    Derulează pagina până la final, repetând până când scrollHeight nu se mai schimbă.
    """
    try:
        previous_height = page.evaluate("() => document.body.scrollHeight")
        while True:
            page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(pause_time)
            new_height = page.evaluate("() => document.body.scrollHeight")
            if new_height == previous_height:
                break
            previous_height = new_height
    except Exception as e:
        logger.error(f"Eroare la auto-scroll: {e}")

# -----------------------
# Așteptarea încărcării imaginilor
# -----------------------
def wait_for_images(page, timeout=5000):
    """
    Așteaptă ca toate imaginile din pagină să fie încărcate complet.
    Se folosește o funcție evaluată în browser care verifică dacă fiecare <img> este complet.
    """
    try:
        page.wait_for_function(
            """() => {
                const imgs = Array.from(document.images);
                return imgs.every(img => img.complete);
            }""",
            timeout=timeout
        )
        logger.info("Toate imaginile au fost încărcate.")
    except TimeoutError:
        logger.warning("Timeout la așteptarea încărcării imaginilor.")

# -----------------------
# Extracția vizuală: screenshot & CSS features
# -----------------------
def get_visual_features(filepath: Path, screenshot_dir: str = "screenshots", server_port: int = 8000, project_root: Path = None):
    """
    Deschide fișierul HTML folosind un server local,
    face un screenshot full-page (după auto-scroll, așteptarea rețelei, a imaginilor și a unui selector cheie)
    și extrage proprietăți CSS extinse de la document.body.
    
    Returnează:
      - features: dicționar cu proprietăți CSS extinse
      - screenshot_path: calea la screenshot sau None
    """
    if project_root is None:
        project_root = Path(os.getcwd())

    try:
        rel_path = filepath.resolve().relative_to(project_root.resolve())
    except Exception as e:
        logger.error(f"Nu s-a putut determina calea relativă pentru {filepath}: {e}")
        return None, None

    url = f"http://localhost:{server_port}/{rel_path.as_posix()}"
    screenshot_dir_path = Path(screenshot_dir)
    screenshot_dir_path.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        
        try:
            # Așteptăm până când nu mai sunt cereri active (networkidle)
            response = page.goto(url, wait_until="networkidle", timeout=15000)
        except Exception as e:
            logger.error(f"Eroare la încărcarea paginii {filepath.name}: {e}")
            browser.close()
            return None, None

        if not response or not response.ok:
            logger.warning(f"Pagina {filepath.name} a returnat un status invalid.")
            browser.close()
            return None, None

        # Așteaptă prezența unui element cheie, de exemplu, #main-content (dacă este prezent)
        try:
            page.wait_for_selector("#main-content", timeout=5000)
        except Exception:
            logger.info(f"Selectorul #main-content nu a fost găsit în {filepath.name}, continuăm totuși.")

        # Auto-scroll pentru a încărca conținutul lazy
        auto_scroll(page, pause_time=0.3)
        # Așteptăm suplimentar pentru ca datele dinamice să se descarce
        time.sleep(2)
        # Așteptăm ca imaginile să se încarce complet
        wait_for_images(page, timeout=5000)

        content = page.content()
        if len(content.strip()) < 200:
            logger.warning(f"Pagina {filepath.name} pare goală sau foarte mică.")
            browser.close()
            return None, None

        lower_content = content.lower()
        error_phrases = ["404", "not found", "internal server error"]
        for err in error_phrases:
            if err in lower_content:
                if len(lower_content.strip()) < 500:
                    logger.warning(f"Pagina {filepath.name} pare a conține eroarea '{err}' și are conținut slab.")
                    browser.close()
                    return None, None
                else:
                    logger.info(f"Pagina {filepath.name} conține '{err}', dar este tratată ca validă (conținut suficient).")
                    break

        # Capturează screenshot-ul complet
        screenshot_path = screenshot_dir_path / f"{filepath.stem}.png"
        try:
            page.screenshot(path=str(screenshot_path), full_page=True)
        except Exception as e:
            logger.error(f"Eroare la salvarea screenshot-ului pentru {filepath.name}: {e}")
            browser.close()
            return None, None
        
        # Extrage proprietăți CSS extinse de la document.body, inclusiv dimensiuni
        try:
            features = page.evaluate("""
                () => {
                    const body = window.getComputedStyle(document.body);
                    const rect = document.body.getBoundingClientRect();
                    return {
                        fontFamily: body.fontFamily,
                        fontSize: body.fontSize,
                        color: body.color,
                        backgroundColor: body.backgroundColor,
                        display: body.display,
                        margin: body.margin,
                        padding: body.padding,
                        lineHeight: body.lineHeight,
                        bodyWidth: rect.width,
                        bodyHeight: rect.height
                    };
                }
            """)
        except Exception as e:
            logger.error(f"Eroare la extragerea proprietăților CSS pentru {filepath.name}: {e}")
            browser.close()
            return None, None

        browser.close()
        logger.info(f"Fișierul {filepath.name} procesat cu succes. Screenshot salvat la {screenshot_path}")
        return features, str(screenshot_path)

# -----------------------
# Compararea screenshot-urilor folosind SSIM
# -----------------------
def compare_screenshots(image_path1: str, image_path2: str) -> float:
    """
    Compară două screenshot-uri folosind Structural Similarity Index (SSIM).
    Returnează un scor între 0 și 1.
    Dacă imaginile sunt prea mari, le redimensionează pentru eficiență.
    """
    try:
        img1 = Image.open(image_path1).convert("L")
        img2 = Image.open(image_path2).convert("L")
    except Exception as e:
        logger.error(f"Eroare la deschiderea imaginilor: {e}")
        return 0.0

    max_dimension = 800
    def resize_if_needed(img):
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            return img.resize(new_size)
        return img

    img1 = resize_if_needed(img1)
    img2 = resize_if_needed(img2)

    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    arr1 = np.array(img1)
    arr2 = np.array(img2)
    try:
        score, _ = ssim(arr1, arr2, full=True)
    except Exception as e:
        logger.error(f"Eroare la calcularea SSIM: {e}")
        return 0.0

    return score

# -----------------------
# Filtrare CSS: elimină proprietățile generice
# -----------------------
def filter_generic_css(features: dict) -> dict:
    """
    Elimină din dicționarul de proprietăți CSS cele considerate generice.
    Ajustează lista după necesitate.
    """
    generic_values = {
        "fontFamily": ["arial", "sans-serif", "helvetica", "times new roman", "serif"],
        "fontSize": ["16px", "14px"],
        "color": ["rgb(0, 0, 0)", "#000000", "black"],
        "backgroundColor": ["rgb(255, 255, 255)", "#ffffff", "white"],
    }
    
    filtered = {}
    for key, value in features.items():
        if key in generic_values:
            value_lower = value.lower()
            if any(gen in value_lower for gen in generic_values[key]):
                continue
        filtered[key] = value
    return filtered

# -----------------------
# Funcție utilitară pentru conversia valorilor CSS în numere
# -----------------------
def parse_css_value(val):
    """
    Încearcă să convertească o valoare CSS (ex: "16px") într-un număr float.
    Dacă nu se poate, returnează None.
    """
    try:
        if isinstance(val, str) and val.endswith("px"):
            return float(val.replace("px", "").strip())
        return float(val)
    except Exception:
        return None

# -----------------------
# Calcul similaritate CSS cu toleranță pentru diferențe minore și ponderi ajustabile
# -----------------------
def simple_css_similarity(f1: dict, f2: dict) -> float:
    """
    Calculează similaritatea CSS bazată pe proprietățile filtrate, cu ponderi.
    Pentru proprietățile numerice, diferențele mai mici de 5% sunt ignorate.
    Returnează scorul de similaritate.
    """
    filtered_f1 = filter_generic_css(f1)
    filtered_f2 = filter_generic_css(f2)
    
    # Ajustează ponderile. Poți modifica valorile după cum dorești.
    weights = {
        "backgroundColor": 2.0,
        "fontSize": 2.0,
        "bodyWidth": 1.0,    # Reducem influența lățimii
        "bodyHeight": 1.0,   # Reducem influența înălțimii
        # Alte proprietăți au greutate implicită 1.0
    }
    
    score_sum = 0.0
    weight_sum = 0.0
    tolerance = 0.05  # toleranță de 5% pentru valorile numerice

    for key, val1 in filtered_f1.items():
        w = weights.get(key, 1.0)
        weight_sum += w
        val2 = filtered_f2.get(key)
        if val2 is None:
            continue
        num1 = parse_css_value(val1)
        num2 = parse_css_value(val2)
        if num1 is not None and num2 is not None:
            avg = (num1 + num2) / 2.0
            if avg == 0:
                match = (num1 == num2)
            else:
                match = abs(num1 - num2) / avg < tolerance
            if match:
                score_sum += w
        else:
            if val1.strip().lower() == val2.strip().lower():
                score_sum += w

    return score_sum / weight_sum if weight_sum else 0.0

# -----------------------
# Acest modul poate fi folosit pentru testare individuală (ex: comenzi de linie)
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculează similaritatea vizuală între două fișiere HTML folosind un server local, Playwright, și comparare de screenshot-uri (SSIM) + CSS extins."
    )
    parser.add_argument("--file1", type=str, required=True, help="Calea către primul fișier HTML")
    parser.add_argument("--file2", type=str, required=True, help="Calea către al doilea fișier HTML")
    parser.add_argument("--screenshot_dir", type=str, default="screenshots", help="Directorul unde se salvează screenshot-urile")
    parser.add_argument("--port", type=int, default=8000, help="Portul pentru serverul HTTP local")
    args = parser.parse_args()

    project_root = Path(os.getcwd())
    httpd = start_http_server(project_root, port=args.port)

    file1 = Path(args.file1)
    file2 = Path(args.file2)

    features1, screenshot1 = get_visual_features(file1, screenshot_dir=args.screenshot_dir, server_port=args.port, project_root=project_root)
    features2, screenshot2 = get_visual_features(file2, screenshot_dir=args.screenshot_dir, server_port=args.port, project_root=project_root)
    
    if features1 is None or features2 is None:
        logger.error("Cel puțin una dintre pagini este invalidă. Nu calculăm similaritatea.")
        httpd.shutdown()
        exit()
    
    css_score = simple_css_similarity(features1, features2)
    image_score = compare_screenshots(screenshot1, screenshot2)
    
    logger.info(f"CSS-based similarity score între {file1.name} și {file2.name}: {css_score:.2f}")
    logger.info(f"Image-based similarity score (SSIM) între {file1.name} și {file2.name}: {image_score:.2f}")
    logger.info(f"Screenshot pentru {file1.name} salvat la: {screenshot1}")
    logger.info(f"Screenshot pentru {file2.name} salvat la: {screenshot2}")

    httpd.shutdown()
    logger.info("Serverul HTTP a fost oprit.")
