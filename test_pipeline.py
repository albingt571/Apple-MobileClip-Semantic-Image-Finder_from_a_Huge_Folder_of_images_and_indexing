"""
Test Script to verify end-to-end embedding, FAISS indexing, SQLite metadata, and search
using MobileCLIP2-S2 on generated sample test images.
"""

import os
from PIL import Image, ImageDraw
from search_engine import SearchEngine


def generate_sample_images(sample_dir: str):
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Black Car
    img_car = Image.new("RGB", (256, 256), color=(20, 20, 20))
    d = ImageDraw.Draw(img_car)
    d.rectangle([40, 100, 216, 170], fill=(5, 5, 5), outline=(100, 100, 100))
    d.ellipse([60, 160, 90, 190], fill=(80, 80, 80))
    d.ellipse([160, 160, 190, 190], fill=(80, 80, 80))
    d.text((50, 110), "BLACK CAR", fill=(220, 220, 220))
    img_car.save(os.path.join(sample_dir, "black_car.jpg"))

    # 2. Red Apple
    img_apple = Image.new("RGB", (256, 256), color=(240, 240, 240))
    d = ImageDraw.Draw(img_apple)
    d.ellipse([60, 60, 196, 200], fill=(210, 20, 20), outline=(150, 10, 10))
    d.text((80, 120), "RED APPLE", fill=(255, 255, 255))
    img_apple.save(os.path.join(sample_dir, "red_apple.png"))

    # 3. Green Tree in Forest
    img_tree = Image.new("RGB", (256, 256), color=(135, 206, 235))
    d = ImageDraw.Draw(img_tree)
    d.rectangle([0, 180, 256, 256], fill=(34, 139, 34))
    d.rectangle([115, 140, 140, 200], fill=(139, 69, 19))
    d.polygon([(128, 40), (70, 150), (186, 150)], fill=(0, 100, 0))
    d.text((70, 90), "GREEN TREE", fill=(255, 255, 255))
    img_tree.save(os.path.join(sample_dir, "green_tree.png"))

    # 4. Corrupt image file for safety test
    with open(os.path.join(sample_dir, "corrupt_file.jpg"), "w") as f:
        f.write("This is not a real image binary data!")

    print(f"Sample test images created in: {sample_dir}")


def main():
    test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_env_data")
    sample_images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_images")

    generate_sample_images(sample_images_dir)

    print("\n--- Initializing Search Engine ---")
    engine = SearchEngine(data_dir=test_data_dir)
    print(f"Device: {engine.device}")

    print("\n--- Running Indexing on Sample Images ---")
    def on_progress(cur, tot, msg):
        print(f"[{cur}/{tot}] {msg}")

    stats = engine.index_folder(sample_images_dir, progress_callback=on_progress)
    print("Indexing Stats:", stats)

    # Queries to test
    queries = ["black car", "red apple", "green forest tree"]

    print("\n--- Running Natural Language Searches ---")
    for q in queries:
        results = engine.search(q, top_k=3)
        print(f"\nQuery: '{q}'")
        for rank, res in enumerate(results, 1):
            fname = os.path.basename(res['file_path'])
            print(f"  Rank {rank}: {fname} | Score: {res['score']:.4f} | Exists: {res['exists']}")


if __name__ == "__main__":
    main()

