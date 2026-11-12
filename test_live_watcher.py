"""
Automated test script to verify:
1. Full folder indexing
2. Real-time live camera / new photo detection, embedding, and FAISS indexing
3. Immediate search retrieval of the newly arrived photo
"""

import os
import time
from PIL import Image, ImageDraw
from search_engine import SearchEngine


def test_live_indexing():
    test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_live_data")
    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_live_folder")
    os.makedirs(sample_dir, exist_ok=True)

    # 1. Populate initial existing image
    img1 = Image.new("RGB", (256, 256), color=(20, 20, 20))
    d1 = ImageDraw.Draw(img1)
    d1.rectangle([50, 100, 200, 150], fill=(0, 0, 0))
    d1.text((60, 110), "BLACK CAR", fill=(255, 255, 255))
    img1.save(os.path.join(sample_dir, "initial_car.jpg"))

    print("\n--- 1. Initializing Search Engine ---")
    engine = SearchEngine(data_dir=test_data_dir)
    print(f"Device: {engine.device}")

    print("\n--- 2. Batch Indexing Existing Folder ---")
    stats = engine.index_folder(sample_dir)
    print("Initial Batch Indexing Stats:", stats)

    # Verify search before new image arrives
    res_before = engine.search("yellow sunflower flower", top_k=1)
    print(f"Search 'yellow sunflower flower' before arrival: {res_before[0]['file_path'] if res_before else 'None'}")

    # 3. Start Live Camera Watcher
    new_images_received = []

    def on_live_ingested(filepath, total_count):
        print(f" [WATCHER EVENT] Ingested live camera photo: {os.path.basename(filepath)} | New total index size: {total_count}")
        new_images_received.append((filepath, total_count))

    print("\n--- 3. Starting Live Camera Watcher ---")
    engine.start_live_watcher(sample_dir, on_new_image_indexed=on_live_ingested)

    # 4. Simulate Camera saving a new image into the folder
    print("\n--- 4. Simulating Camera Snapshot: Saving 'camera_sunflower.jpg' ---")
    time.sleep(1)
    img2 = Image.new("RGB", (256, 256), color=(255, 255, 240))
    d2 = ImageDraw.Draw(img2)
    d2.ellipse([50, 50, 200, 200], fill=(255, 215, 0))
    d2.ellipse([100, 100, 150, 150], fill=(139, 69, 19))
    d2.text((70, 110), "SUNFLOWER", fill=(0, 0, 0))
    img2.save(os.path.join(sample_dir, "camera_sunflower.jpg"))

    # Wait for watchdog and worker queue to index
    timeout = 10
    start = time.time()
    while len(new_images_received) == 0 and time.time() - start < timeout:
        time.sleep(0.5)

    print("\n--- 5. Verifying Search Results Immediately After Camera Snapshot ---")
    res_after = engine.search("yellow sunflower flower", top_k=2)
    for rank, r in enumerate(res_after, 1):
        print(f"Rank {rank}: {os.path.basename(r['file_path'])} | Score: {r['score']:.4f}")

    engine.stop_live_watcher()
    assert any("camera_sunflower.jpg" in r['file_path'] for r in res_after[:1]), "Camera image was not top ranked!"
    print("\n SUCCESS: Live camera image was automatically indexed and instantly searchable!")


if __name__ == "__main__":
    test_live_indexing()

