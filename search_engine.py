"""
Sensie Image Search - Core Search and Embedding Engine.
Utilizes MobileCLIP2-S2 (dfndr2b) via open-clip-torch, FAISS vector index, and SQLite.
"""

import os
import sqlite3
import time
from typing import List, Tuple, Dict, Any, Optional, Callable
from PIL import Image, UnidentifiedImageError
import numpy as np
import torch
import open_clip
import faiss

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
        self._init_db()
        self._load_or_create_index()

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
        if os.path.exists(self.faiss_index_path):
            try:
                self.faiss_index = faiss.read_index(self.faiss_index_path)
                return
            except Exception as e:
                print(f"Warning: Failed to load existing FAISS index ({e}), creating a fresh index.")
        # IndexFlatIP uses Inner Product (Cosine similarity when vectors are L2-normalized)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

    def save_index(self):
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, self.faiss_index_path)

    def load_model(self, status_callback: Optional[Callable[[str], None]] = None):
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

        if need_rebuild:
            if progress_callback:
                progress_callback(0, len(disk_files_map), "Re-synchronizing database and index...")
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                if deleted_paths:
                    cursor.executemany("DELETE FROM images WHERE file_path = ?", [(p,) for p in deleted_paths])
                conn.commit()

            # We will re-index all files currently in folder_path that need re-embedding
            to_index = list(disk_files_map.keys())
            # Reset index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM images WHERE folder_path = ?", (folder_path,))
                conn.commit()

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
                        # Convert to RGB safely
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
                    # L2 normalize
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    embeddings = image_features.cpu().numpy().astype(np.float32)

                # Add to FAISS
                self.faiss_index.add(embeddings)

                # Add to SQLite
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

        self.save_index()

        # Summary
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

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Encodes query string, searches FAISS, and retrieves matching image paths and scores.
        """
        query = query.strip()
        if not query:
            return []

        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []

        self.load_model()

        # Encode query text
        with torch.no_grad():
            text_tokens = self.tokenizer([query]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            query_vector = text_features.cpu().numpy().astype(np.float32)

        # Search FAISS
        k = min(top_k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_vector, k)

        scores = scores[0]
        indices = indices[0]

        # Retrieve file_paths by row id offset in images table
        # Since FAISS indices correspond 1:1 with sequential rows in images table:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Fetch all rows ordered by id
            cursor.execute("SELECT id, file_path, folder_path FROM images ORDER BY id ASC")
            all_records = cursor.fetchall()

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(all_records):
                continue
            rec_id, file_path, folder_path = all_records[idx]
            # Verify file exists on disk
            exists = os.path.exists(file_path)
            results.append({
                "id": rec_id,
                "file_path": file_path,
                "folder_path": folder_path,
                "score": float(score),
                "exists": exists
            })

        return results

