import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import os
import platform
import time
import shutil
import multiprocessing
import threading
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".wmv", ".flv", ".mpeg", ".mpg"}

class MediaToMKVConverter(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg="#2e2e2e")
        self.master = master
        self.master.title("Media to MKV Converter")
        self.master.geometry("500x550")
        self.pack(fill=tk.BOTH, expand=True)

        self.cancel_flag = threading.Event()
        self.files = []

        self.ffmpeg_path = shutil.which('ffmpeg')
        if not self.ffmpeg_path:
            messagebox.showerror("Error", "ffmpeg not found in PATH.")
            master.destroy()
            return

        self._build_ui()

    def _build_ui(self):
        tk.Label(self, text="Select media files to convert to MKV", bg="#2e2e2e", fg="white").pack(pady=10)
        tk.Button(self, text="Select Files", command=self.select_files).pack(pady=5)

        opts = tk.Frame(self, bg="#2e2e2e")
        self.delete_var = tk.BooleanVar()
        tk.Checkbutton(opts, text="Delete originals", variable=self.delete_var,
                       bg="#2e2e2e", fg="white", selectcolor="#2e2e2e").pack(side=tk.LEFT, padx=5)
        self.same_var = tk.BooleanVar()
        tk.Checkbutton(opts, text="Same folder output", variable=self.same_var,
                       bg="#2e2e2e", fg="white", selectcolor="#2e2e2e").pack(side=tk.LEFT, padx=5)
        opts.pack(pady=5)

        self.progress = ttk.Progressbar(self, orient="horizontal", mode="determinate", length=400)
        self.progress.pack(pady=10)

        self.status = tk.Label(self, text="Idle", bg="#2e2e2e", fg="white")
        self.status.pack(pady=5)

        btns = tk.Frame(self, bg="#2e2e2e")
        self.start_btn = tk.Button(btns, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.cancel_btn = tk.Button(btns, text="Cancel", command=self.cancel, state=tk.DISABLED)
        self.cancel_btn.pack(side=tk.LEFT, padx=5)
        btns.pack(pady=5)

        self.log_text = tk.Text(self, height=10, bg="#1e1e1e", fg="lime", state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=10)

    def select_files(self):
        paths = filedialog.askopenfilenames(
            filetypes=[("Media", "*" + " *.".join(ext.lstrip('.') for ext in SUPPORTED_EXTENSIONS))]
        )
        self.files = [Path(p) for p in paths if Path(p).suffix.lower() in SUPPORTED_EXTENSIONS]
        self.status.config(text=f"{len(self.files)} files ready")

    def start(self):
        if not self.files:
            messagebox.showwarning("No files", "Select at least one media file.")
            return
        self.cancel_flag.clear()
        self.start_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        threading.Thread(target=self._process).start()

    def cancel(self):
        self.cancel_flag.set()
        self.status.config(text="Cancelling...")

    def _process(self):
        total = len(self.files)
        self.progress.config(maximum=total, value=0)
        stats = {'input': 0, 'output': 0, 'converted': 0, 'failed': []}
        formats = {}
        start = time.time()

        base_out = Path.cwd() / "Converted_MKV"
        base_out.mkdir(exist_ok=True)

        for idx, src in enumerate(self.files, 1):
            if self.cancel_flag.is_set(): break
            ext = src.suffix.lower()
            formats[ext] = formats.get(ext, 0) + 1
            out_dir = src.parent if self.same_var.get() else base_out
            out = out_dir / f"{src.stem}.mkv"
            out_dir.mkdir(parents=True, exist_ok=True)

            self.status.config(text=f"[{idx}/{total}] {src.name}")
            cmd = [self.ffmpeg_path, "-y", "-i", str(src),
                   "-map", "0:v", "-map", "0:a", "-c", "copy", str(out)]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            err = res.stderr.decode(errors='ignore').strip().splitlines()
            if res.returncode == 0 and out.exists() and out.stat().st_size > 1024:
                stats['converted'] += 1
                stats['input'] += src.stat().st_size
                stats['output'] += out.stat().st_size
                if self.delete_var.get(): src.unlink()
                self._log(f"OK: {src.name}")
            else:
                stats['failed'].append((src.name, err[-1] if err else 'Unknown error'))
                self._log(f"FAIL: {src.name} -> {err[-1] if err else 'Unknown'}")

            self.progress.step()

        duration = time.time() - start
        cpu = platform.processor() or platform.uname().processor
        cores = multiprocessing.cpu_count()
        free_gb = shutil.disk_usage(Path.cwd()).free / (1024**3)

        summary = [
            f"Converted: {stats['converted']}/{total}",
            f"Size in: {stats['input']/(1024**2):.2f} MB, out: {stats['output']/(1024**2):.2f} MB",
            f"Formats: {', '.join(f'{k}({v})' for k,v in formats.items())}",
            f"CPU: {cpu} ({cores} cores)",
            f"Free disk: {free_gb:.2f} GB",
            f"Time: {duration:.2f}s"
        ]
        if stats['failed']:
            summary.append("Failures:")
            summary += [f" {n}: {e}" for n, e in stats['failed']]
        summary_text = "\n".join(summary)

        self.status.config(text="Done")
        self._log("\nSummary:\n" + summary_text)
        messagebox.showinfo("Summary", summary_text)

        target = self.files[0].parent if self.same_var.get() else base_out
        os.startfile(target) if os.name=='nt' else subprocess.run(["open", str(target)])

        self.start_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)

    def _log(self, line):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

if __name__ == '__main__':
    root = tk.Tk()
    MediaToMKVConverter(root)
    root.mainloop()
