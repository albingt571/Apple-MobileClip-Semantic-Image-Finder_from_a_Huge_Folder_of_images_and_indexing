"""
Sensie Image Search - Core Search, Live Monitoring, and Embedding Engine.
Utilizes MobileCLIP2-S2 (dfndr2b) via open-clip-torch, FAISS vector index, SQLite,
and live background directory monitoring via watchdog.
"""

import os
import sqlite3
import time
import threading
import queue
from typing import List, Tuple, Dict, Any, Optional, Callable
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
import open_clip
import faiss
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Supported image file extensions
SUPPORTED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'
}

DEFAULT_MODEL_NAME = "MobileCLIP2-S2"
DEFAULT_PRETRAINED = "dfndr2b"


class SearchEngine:
    def __init__(self, data_dir: str, device: Optional[str] = None):
        """
        data_dir: directory where sqlite db and faiss index files will be stored.
        device: 'cuda' or 'cpu'. If None, auto-selects cuda if available, else cpu.
        """
        self.data_dir = os.path.abspath(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)

        self.db_path = os.path.join(self.data_dir, "metadata.db")
        self.faiss_index_path = os.path.join(self.data_dir, "faiss.index")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"

        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.embedding_dim = 512

        self.faiss_index = None
        self.lock = threading.Lock()

        self._init_db()
        self._load_or_create_index()

        # Live folder watcher state
        self.observer = None
        self.watched_folder = None
        self.live_queue = queue.Queue()
        self.live_worker_thread = None
        self.stop_live_event = threading.Event()
        self.live_callbacks = []

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    folder_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    mtime REAL NOT NULL,
                    indexed_at REAL NOT NULL,
                    status TEXT DEFAULT 'ok'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_folder ON images(folder_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_filepath ON images(file_path)")
            conn.commit()

    def _load_or_create_index(self):
        with self.lock:
            if os.path.exists(self.faiss_index_path):
                try:
                    self.faiss_index = faiss.read_index(self.faiss_index_path)
                    return
                except Exception as e:
                    print(f"Warning: Failed to load existing FAISS index ({e}), creating a fresh index.")
            # IndexFlatIP uses Inner Product (Cosine similarity when vectors are L2-normalized)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

    def save_index(self):
        with self.lock:
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, self.faiss_index_path)

    def load_model(self, status_callback: Optional[Callable[[str], None]] = None):
        if self.model is not None:
            return

        with self.lock:
            if self.model is not None:
                return

            if status_callback:
                status_callback(f"Loading {DEFAULT_MODEL_NAME} ({DEFAULT_PRETRAINED}) on {self.device.upper()}...")

            # Load MobileCLIP2-S2 with open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                DEFAULT_MODEL_NAME,
                pretrained=DEFAULT_PRETRAINED,
                device=self.device
            )
            tokenizer = open_clip.get_tokenizer(DEFAULT_MODEL_NAME)

            model.eval()
            self.model = model
            self.preprocess = preprocess
            self.tokenizer = tokenizer

            # Check embedding dimension
            try:
                with torch.no_grad():
                    dummy_tokens = self.tokenizer(["test"]).to(self.device)
                    dummy_emb = self.model.encode_text(dummy_tokens)
                    self.embedding_dim = dummy_emb.shape[-1]
                    if self.faiss_index.d != self.embedding_dim:
                        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            except Exception:
                pass

            if status_callback:
                status_callback(f"Model {DEFAULT_MODEL_NAME} ready ({self.device.upper()}).")

    def scan_folder(self, folder_path: str) -> List[str]:
        """Recursively scan for supported image files."""
        folder_path = os.path.abspath(folder_path)
        valid_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    valid_files.append(os.path.join(root, file))
        return valid_files

    def _wait_for_file_ready(self, file_path: str, timeout: float = 3.0, interval: float = 0.2) -> bool:
        """Wait until a newly created camera file is fully written and unlocked."""
        start = time.time()
        last_size = -1
        while time.time() - start < timeout:
            if not os.path.exists(file_path):
                return False
            try:
                curr_size = os.path.getsize(file_path)
                if curr_size > 0 and curr_size == last_size:
                    # Try opening to confirm no write lock
                    with open(file_path, "rb"):
                        pass
                    return True
                last_size = curr_size
            except (OSError, PermissionError):
                pass
            time.sleep(interval)
        return os.path.exists(file_path) and os.path.getsize(file_path) > 0

    def index_single_image(self, file_path: str, folder_path: str) -> bool:
        """Embed and append a single newly arrived image (e.g. from camera/watchdog)."""
        file_path = os.path.abspath(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return False

        if not self._wait_for_file_ready(file_path):
            return False

        self.load_model()

        try:
            stat = os.stat(file_path)
            file_size, mtime = stat.st_size, stat.st_mtime
        except OSError:
            return False

        # Check if already present and unchanged
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, file_size, mtime FROM images WHERE file_path = ?", (file_path,))
            row = cursor.fetchone()
            if row and row[1] == file_size and abs(row[2] - mtime) < 1e-4:
                return False

        # If modified existing image, rebuild full folder to keep FAISS alignment intact
        if row:
            self.index_folder(folder_path)
            return True

        # Process image
        try:
            with Image.open(file_path) as img:
                rgb_img = img.convert("RGB")
                tensor = self.preprocess(rgb_img).unsqueeze(0).to(self.device)
        except (UnidentifiedImageError, OSError, ValueError):
            return False

        with self.lock:
            with torch.no_grad():
                features = self.model.encode_image(tensor)
                features /= features.norm(dim=-1, keepdim=True)
                embedding = features.cpu().numpy().astype(np.float32)

            self.faiss_index.add(embedding)

            now = time.time()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO images (file_path, folder_path, file_size, mtime, indexed_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (file_path, folder_path, file_size, mtime, now))
                conn.commit()

            faiss.write_index(self.faiss_index, self.faiss_index_path)

        return True

    def index_folder(
        self,
        folder_path: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None
    ) -> Dict[str, Any]:
        """
        Performs incremental indexing for folder_path:
        - Detects new and modified files
        - Removes deleted/missing files
        - Embeds in batches and updates FAISS index + SQLite DB
        """
        self.load_model()
        folder_path = os.path.abspath(folder_path)

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")

        disk_files = self.scan_folder(folder_path)
        disk_files_map = {}
        for f in disk_files:
            try:
                stat = os.stat(f)
                disk_files_map[f] = (stat.st_size, stat.st_mtime)
            except OSError:
                continue

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, file_path, file_size, mtime FROM images WHERE folder_path = ?", (folder_path,))
            db_records = {row[1]: (row[0], row[2], row[3]) for row in cursor.fetchall()}

        # 1. Detect missing/deleted files
        deleted_paths = [p for p in db_records.keys() if p not in disk_files_map]

        # 2. Detect files needing indexing (new or modified)
        to_index = []
        for p, (size, mtime) in disk_files_map.items():
            if p not in db_records:
                to_index.append(p)
            else:
                _, old_size, old_mtime = db_records[p]
                if size != old_size or abs(mtime - old_mtime) > 1e-4:
                    to_index.append(p)

        # If any files were deleted or modified, we rebuild the index for consistency
        need_rebuild = len(deleted_paths) > 0 or any(p in db_records for p in to_index)

        with self.lock:
            if need_rebuild:
                if progress_callback:
                    progress_callback(0, len(disk_files_map), "Re-synchronizing database and index...")
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    if deleted_paths:
                        cursor.executemany("DELETE FROM images WHERE file_path = ?", [(p,) for p in deleted_paths])
                    cursor.execute("DELETE FROM images WHERE folder_path = ?", (folder_path,))
                    conn.commit()

                # Re-index all files currently in folder_path
                to_index = list(disk_files_map.keys())
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

            total = len(to_index)
            indexed_count = 0
            corrupt_count = 0
            batch_size = 16

            for i in range(0, total, batch_size):
                if should_cancel and should_cancel():
                    if progress_callback:
                        progress_callback(i, total, "Indexing cancelled by user.")
                    break

                batch_paths = to_index[i:i + batch_size]
                valid_batch_images = []
                valid_batch_paths = []

                for path in batch_paths:
                    try:
                        with Image.open(path) as img:
                            rgb_img = img.convert("RGB")
                            tensor = self.preprocess(rgb_img)
                            valid_batch_images.append(tensor)
                            valid_batch_paths.append(path)
                    except (UnidentifiedImageError, OSError, ValueError):
                        corrupt_count += 1
                        continue

                if valid_batch_images:
                    batch_tensor = torch.stack(valid_batch_images).to(self.device)
                    with torch.no_grad():
                        image_features = self.model.encode_image(batch_tensor)
                        image_features /= image_features.norm(dim=-1, keepdim=True)
                        embeddings = image_features.cpu().numpy().astype(np.float32)

                    self.faiss_index.add(embeddings)

                    now = time.time()
                    with sqlite3.connect(self.db_path) as conn:
                        cursor = conn.cursor()
                        for p in valid_batch_paths:
                            sz, mt = disk_files_map.get(p, (0, 0))
                            cursor.execute("""
                                INSERT INTO images (file_path, folder_path, file_size, mtime, indexed_at)
                                VALUES (?, ?, ?, ?, ?)
                            """, (p, folder_path, sz, mt, now))
                        conn.commit()

                    indexed_count += len(valid_batch_paths)

                if progress_callback:
                    progress_callback(min(i + batch_size, total), total, f"Processed {min(i + batch_size, total)} of {total} images...")

            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, self.faiss_index_path)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM images WHERE folder_path = ?", (folder_path,))
            total_active = cursor.fetchone()[0]

        return {
            "folder": folder_path,
            "total_files_found": len(disk_files),
            "new_or_updated_indexed": indexed_count,
            "corrupt_or_skipped": corrupt_count,
            "deleted_removed": len(deleted_paths),
            "total_in_index": total_active
        }

    def start_live_watcher(self, folder_path: str, on_new_image_indexed: Optional[Callable[[str, int], None]] = None):
        """
        Starts a background filesystem watcher (watchdog) on folder_path.
        Automatically detects, waits for camera write locks, and embeds incoming images.
        """
        self.stop_live_watcher()
        folder_path = os.path.abspath(folder_path)
        self.watched_folder = folder_path
        self.stop_live_event.clear()

        if on_new_image_indexed:
            self.live_callbacks = [on_new_image_indexed]
        else:
            self.live_callbacks = []

        class ImageFileHandler(FileSystemEventHandler):
            def __init__(self, target_queue):
                super().__init__()
                self.target_queue = target_queue

            def on_created(self, event):
                if not event.is_directory:
                    ext = os.path.splitext(event.src_path)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        self.target_queue.put(event.src_path)

            def on_moved(self, event):
                if not event.is_directory:
                    ext = os.path.splitext(event.dest_path)[1].lower()
                    if ext in SUPPORTED_EXTENSIONS:
                        self.target_queue.put(event.dest_path)

        def worker():
            while not self.stop_live_event.is_set():
                try:
                    filepath = self.live_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if self.stop_live_event.is_set():
                    break

                try:
                    success = self.index_single_image(filepath, self.watched_folder)
                    if success:
                        with sqlite3.connect(self.db_path) as conn:
                            cur = conn.cursor()
                            cur.execute("SELECT COUNT(*) FROM images WHERE folder_path = ?", (self.watched_folder,))
                            count = cur.fetchone()[0]

                        for cb in self.live_callbacks:
                            try:
                                cb(filepath, count)
                            except Exception:
                                pass
                except Exception as e:
                    print(f"Error indexing live incoming image '{filepath}': {e}")
                finally:
                    self.live_queue.task_done()

        self.live_worker_thread = threading.Thread(target=worker, daemon=True)
        self.live_worker_thread.start()

        event_handler = ImageFileHandler(self.live_queue)
        self.observer = Observer()
        self.observer.schedule(event_handler, folder_path, recursive=True)
        self.observer.start()

    def stop_live_watcher(self):
        """Stops the active live directory watcher."""
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=2)
            except Exception:
                pass
            self.observer = None

        self.stop_live_event.set()
        if self.live_worker_thread and self.live_worker_thread.is_alive():
            self.live_worker_thread.join(timeout=2)
        self.live_worker_thread = None
        self.watched_folder = None

    def search(self, query: str, top_k: int = 24) -> List[Dict[str, Any]]:
        """
        Encodes query string, searches FAISS, and retrieves matching image paths and scores.
        """
        query = query.strip()
        if not query:
            return []

        with self.lock:
            if self.faiss_index is None or self.faiss_index.ntotal == 0:
                return []

        self.load_model()

        with torch.no_grad():
            text_tokens = self.tokenizer([query]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            query_vector = text_features.cpu().numpy().astype(np.float32)

        with self.lock:
            k = min(top_k, self.faiss_index.ntotal)
            scores, indices = self.faiss_index.search(query_vector, k)

            scores = scores[0]
            indices = indices[0]

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, file_path, folder_path FROM images ORDER BY id ASC")
                all_records = cursor.fetchall()

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(all_records):
                continue
            rec_id, file_path, folder_path = all_records[idx]
            exists = os.path.exists(file_path)
            results.append({
                "id": rec_id,
                "file_path": file_path,
                "folder_path": folder_path,
                "score": float(score),
                "exists": exists
            })

        return results
