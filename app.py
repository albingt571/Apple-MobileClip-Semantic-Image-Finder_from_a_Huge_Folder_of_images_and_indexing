"""
Sensie Image Search - Modern Desktop GUI (Tkinter + Canvas).
Features:
- Folder picker & status reporting
- Background threading for batch indexing & natural search
- Real-time live camera / new image watchdog listener (auto-indexes incoming photos)
- Thumbnail gallery grid with scores and click-to-open
- Progress bar and friendly error/status feedback
"""

import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from search_engine import SearchEngine, DEFAULT_MODEL_NAME


class ImageSearchApp:
    def __init__(self, root: tk.Tk, data_dir: str):
        self.root = root
        self.root.title("Apple MobileCLIP2 Semantic Image Search & Live Camera Indexer")
        self.root.geometry("1060x740")
        self.root.minsize(840, 600)

        self.data_dir = data_dir
        self.engine = SearchEngine(data_dir=data_dir)

        self.selected_folder = tk.StringVar(value="")
        self.search_query = tk.StringVar(value="")
        self.device_var = tk.StringVar(value=self.engine.device.upper())
        self.status_var = tk.StringVar(value="Ready. Select an image folder to get started.")
        self.live_watch_var = tk.BooleanVar(value=True)

        self.indexing_thread = None
        self.thumbnail_cache = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        # Top Frame: Folder Selection & Device Info
        top_frame = ttk.LabelFrame(self.root, text=" 1. Folder Selection & Live Camera Ingestion ", padding=10)
        top_frame.pack(fill="x", padx=12, pady=6)

        ttk.Label(top_frame, text="Folder:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.folder_entry = ttk.Entry(top_frame, textvariable=self.selected_folder, width=54)
        self.folder_entry.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        top_frame.columnconfigure(1, weight=1)

        browse_btn = ttk.Button(top_frame, text="Browse...", command=self.on_browse_folder)
        browse_btn.grid(row=0, column=2, padx=4, pady=4)

        self.index_btn = ttk.Button(top_frame, text="Index Folder", command=self.on_start_indexing)
        self.index_btn.grid(row=0, column=3, padx=4, pady=4)

        # Options Row
        options_frame = ttk.Frame(top_frame)
        options_frame.grid(row=1, column=1, columnspan=3, sticky="w", pady=2)

        self.watch_chk = ttk.Checkbutton(
            options_frame,
            text="Auto-index incoming camera / new photos in real time",
            variable=self.live_watch_var,
            command=self.on_toggle_live_watch
        )
        self.watch_chk.pack(side="left", padx=4)

        self.live_indicator = ttk.Label(options_frame, text="● Live Watcher Idle", foreground="#666666", font=("Segoe UI", 9, "bold"))
        self.live_indicator.pack(side="left", padx=12)

        device_lbl = ttk.Label(options_frame, text=f"Runtime: {self.engine.device.upper()} | Model: {DEFAULT_MODEL_NAME}", foreground="#444")
        device_lbl.pack(side="right", padx=4)

        # Middle Frame: Natural Language Query
        query_frame = ttk.LabelFrame(self.root, text=" 2. Semantic Natural Language Search ", padding=10)
        query_frame.pack(fill="x", padx=12, pady=4)

        ttk.Label(query_frame, text="Search:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.query_entry = ttk.Entry(query_frame, textvariable=self.search_query, width=60)
        self.query_entry.grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        self.query_entry.bind("<Return>", lambda event: self.on_search())
        query_frame.columnconfigure(1, weight=1)

        search_btn = ttk.Button(query_frame, text="Search", command=self.on_search)
        search_btn.grid(row=0, column=2, padx=4, pady=4)

        # Progress bar
        self.progress_bar = ttk.Progressbar(self.root, mode="determinate")
        self.progress_bar.pack(fill="x", padx=12, pady=2)

        # Status Bar
        status_frame = ttk.Frame(self.root, padding=(12, 2))
        status_frame.pack(fill="x")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, foreground="#2b4c7e", font=("Segoe UI", 9, "italic"))
        self.status_label.pack(side="left")

        # Results Area (Scrollable Canvas)
        results_container = ttk.LabelFrame(self.root, text=" Search Results ", padding=8)
        results_container.pack(fill="both", expand=True, padx=12, pady=6)

        self.canvas = tk.Canvas(results_container, borderwidth=0, background="#f5f6f8")
        self.scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(xscrollcommand=None, yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.bind('<Configure>', self._on_canvas_configure)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def on_browse_folder(self):
        folder = filedialog.askdirectory(title="Select Folder of Images")
        if folder:
            self.selected_folder.set(folder)
            self.status_var.set(f"Selected folder: {folder}. Click 'Index Folder' to process.")

    def on_toggle_live_watch(self):
        folder = self.selected_folder.get().strip()
        if not self.live_watch_var.get():
            self.engine.stop_live_watcher()
            self.live_indicator.config(text="● Live Watcher Paused", foreground="#888888")
        elif folder and os.path.isdir(folder):
            self._start_live_monitoring(folder)

    def _start_live_monitoring(self, folder: str):
        if not self.live_watch_var.get():
            return

        def on_live_ingest(filepath, total_count):
            filename = os.path.basename(filepath)
            self.root.after(0, lambda: self._on_live_photo_indexed(filename, total_count))

        self.engine.start_live_watcher(folder, on_new_image_indexed=on_live_ingest)
        self.live_indicator.config(text="● Live Watcher Active (Monitoring Camera/New Photos)", foreground="#007a33")

    def _on_live_photo_indexed(self, filename: str, total_count: int):
        self.status_var.set(f"📷 Live Auto-Indexed new photo: '{filename}' (Total in Index: {total_count})")
        # If user has an active query, auto-refresh search to reflect new photo
        active_q = self.search_query.get().strip()
        if active_q:
            self.on_search(silent=True)

    def on_start_indexing(self):
        folder = self.selected_folder.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Invalid Folder", "Please select a valid image folder first.")
            return

        self.index_btn.config(state="disabled")
        self.progress_bar["value"] = 0
        self.status_var.set("Indexing existing images in folder...")

        def run():
            try:
                def progress(current, total, msg):
                    pct = (current / total * 100) if total > 0 else 0
                    self.root.after(0, lambda: self._update_progress(pct, msg))

                result = self.engine.index_folder(folder, progress_callback=progress)
                msg = (f"Finished indexing! Added/updated: {result['new_or_updated_indexed']}, "
                       f"Corrupt/Skipped: {result['corrupt_or_skipped']}, "
                       f"Total in Index: {result['total_in_index']}.")
                self.root.after(0, lambda: self._finish_indexing(msg, folder))
            except Exception as e:
                err_msg = f"Indexing failed: {str(e)}"
                self.root.after(0, lambda: self._finish_indexing(err_msg, folder, is_error=True))

        self.indexing_thread = threading.Thread(target=run, daemon=True)
        self.indexing_thread.start()

    def _update_progress(self, percent: float, msg: str):
        self.progress_bar["value"] = percent
        self.status_var.set(msg)

    def _finish_indexing(self, message: str, folder: str, is_error: bool = False):
        self.index_btn.config(state="normal")
        self.progress_bar["value"] = 100 if not is_error else 0
        self.status_var.set(message)
        if is_error:
            messagebox.showerror("Indexing Error", message)
        else:
            self._start_live_monitoring(folder)
            messagebox.showinfo("Indexing Complete", f"{message}\n\nLive camera / incoming image monitoring is active.")

    def on_search(self, silent: bool = False):
        query = self.search_query.get().strip()
        if not query:
            if not silent:
                self.status_var.set("Please enter a search query.")
            return

        if not silent:
            self.status_var.set(f"Searching for '{query}'...")
            self.progress_bar["value"] = 50

        def run():
            try:
                results = self.engine.search(query, top_k=24)
                self.root.after(0, lambda: self._render_results(results, query))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(f"Search error: {str(e)}"))
                self.root.after(0, lambda: self.progress_bar.configure(value=0))

        threading.Thread(target=run, daemon=True).start()

    def _render_results(self, results, query: str):
        self.progress_bar["value"] = 100
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.thumbnail_cache.clear()

        if not results:
            self.status_var.set(f"No matching images found for '{query}'. Make sure images are indexed.")
            empty_lbl = ttk.Label(self.scrollable_frame, text=f"No results found for \"{query}\".\nEnsure images are indexed or folder is selected.", font=("Segoe UI", 11), padding=20)
            empty_lbl.pack()
            return

        self.status_var.set(f"Found {len(results)} matching images for '{query}'.")

        columns = 4
        thumb_size = (180, 140)

        for idx, item in enumerate(results):
            r = idx // columns
            c = idx % columns

            card = ttk.Frame(self.scrollable_frame, relief="ridge", padding=5)
            card.grid(row=r, column=c, padx=8, pady=8, sticky="nsew")
            self.scrollable_frame.columnconfigure(c, weight=1)

            file_path = item["file_path"]
            score = item["score"]
            file_name = os.path.basename(file_path)

            try:
                with Image.open(file_path) as img:
                    img.thumbnail(thumb_size)
                    photo = ImageTk.PhotoImage(img)
                    self.thumbnail_cache.append(photo)
                    img_label = ttk.Label(card, image=photo, cursor="hand2")
                    img_label.pack(pady=2)
                    img_label.bind("<Button-1>", lambda e, p=file_path: self._open_image(p))
            except Exception:
                img_label = ttk.Label(card, text="[Thumbnail Unavailable]", width=20, anchor="center")
                img_label.pack(pady=20)

            name_lbl = ttk.Label(card, text=file_name, font=("Segoe UI", 8, "bold"), wraplength=170)
            name_lbl.pack(anchor="w")

            score_lbl = ttk.Label(card, text=f"Match Score: {score:.3f}", font=("Segoe UI", 8), foreground="#555")
            score_lbl.pack(anchor="w")

            btn_open = ttk.Button(card, text="Open File", command=lambda p=file_path: self._open_image(p))
            btn_open.pack(fill="x", pady=2)

    def _open_image(self, file_path: str):
        if not os.path.exists(file_path):
            messagebox.showwarning("File Not Found", f"The file no longer exists at:\n{file_path}")
            return
        try:
            if sys.platform == "win32":
                os.startfile(file_path)
            else:
                subprocess.run(["xdg-open", file_path])
        except Exception as e:
            messagebox.showerror("Error Opening File", str(e))

    def on_close(self):
        try:
            self.engine.stop_live_watcher()
        except Exception:
            pass
        self.root.destroy()


def main():
    app_data_dir = r"E:\SENSIE_AGENTS" if os.path.exists(r"E:\SENSIE_AGENTS") else os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensie_data")
    os.makedirs(app_data_dir, exist_ok=True)

    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")

    app = ImageSearchApp(root, data_dir=app_data_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
