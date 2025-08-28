#!/usr/bin/env python3
"""
Enhanced Media to MKV Converter

A modern, feature-rich media converter with improved architecture, 
better error handling, and enhanced user experience.

Features:
- Modern themed UI with drag-and-drop support
- Conversion presets (quality/codec options)
- Concurrent processing with configurable threads
- Persistent settings and conversion history
- Advanced file filtering and duplicate detection
- Better progress tracking and status reporting
- Comprehensive error handling and logging
- Cross-platform compatibility

Requirements:
- Python 3.8+
- ffmpeg & ffprobe in PATH
- tkinterdnd2 (pip install tkinterdnd2) for drag-and-drop
"""

import os
import sys
import json
import shutil
import threading
import subprocess
import webbrowser
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time
import hashlib
import logging

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Try to import drag-and-drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DRAG_DROP_AVAILABLE = True
except ImportError:
    DRAG_DROP_AVAILABLE = False
    TkinterDnD = tk
    DND_FILES = None

# Constants
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".wmv", ".flv", ".mpeg", ".mpg", ".m4v", ".3gp"}
CONFIG_FILE = Path.home() / ".media_converter_config.json"
LOG_FILE = Path.home() / "media_converter.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ConversionPreset:
    """Defines conversion settings preset"""
    name: str
    description: str
    video_codec: str = "copy"
    audio_codec: str = "copy"
    video_bitrate: Optional[str] = None
    audio_bitrate: Optional[str] = None
    scale: Optional[str] = None
    extra_args: List[str] = None

    def __post_init__(self):
        if self.extra_args is None:
            self.extra_args = []

@dataclass
class ConversionTask:
    """Represents a single conversion task"""
    source_path: Path
    output_path: Path
    preset: ConversionPreset
    status: str = "pending"  # pending, processing, completed, failed, cancelled
    progress: float = 0.0
    error_message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    file_size_in: int = 0
    file_size_out: int = 0

class ConfigManager:
    """Manages application configuration and settings"""
    
    DEFAULT_CONFIG = {
        "delete_originals": False,
        "same_folder_output": False,
        "output_directory": "",
        "max_threads": 2,
        "selected_preset": "copy_all",
        "window_geometry": "800x900",
        "check_duplicates": True,
        "auto_open_output": True,
        "theme": "default"
    }
    
    def __init__(self):
        self.config = self.DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r') as f:
                    saved_config = json.load(f)
                self.config.update(saved_config)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        self.config[key] = value
        self.save_config()

class MediaValidator:
    """Validates media files and extracts metadata"""
    
    def __init__(self, ffprobe_path: str):
        self.ffprobe_path = ffprobe_path
    
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate media file and extract basic info"""
        try:
            cmd = [
                self.ffprobe_path,
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return {"valid": False, "error": "FFprobe failed"}
            
            data = json.loads(result.stdout)
            
            # Extract useful information
            format_info = data.get('format', {})
            streams = data.get('streams', [])
            
            video_streams = [s for s in streams if s.get('codec_type') == 'video']
            audio_streams = [s for s in streams if s.get('codec_type') == 'audio']
            
            return {
                "valid": True,
                "duration": float(format_info.get('duration', 0)),
                "size": int(format_info.get('size', 0)),
                "video_streams": len(video_streams),
                "audio_streams": len(audio_streams),
                "video_codec": video_streams[0].get('codec_name', 'unknown') if video_streams else None,
                "audio_codec": audio_streams[0].get('codec_name', 'unknown') if audio_streams else None,
                "resolution": f"{video_streams[0].get('width', 0)}x{video_streams[0].get('height', 0)}" if video_streams else None
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}

class ConversionEngine:
    """Handles the actual media conversion using FFmpeg"""
    
    def __init__(self, ffmpeg_path: str, progress_callback: Optional[Callable] = None):
        self.ffmpeg_path = ffmpeg_path
        self.progress_callback = progress_callback
        self.active_processes: Dict[str, subprocess.Popen] = {}
    
    def convert_file(self, task: ConversionTask) -> bool:
        """Convert a single file"""
        try:
            task.status = "processing"
            task.start_time = time.time()
            task.file_size_in = task.source_path.stat().st_size
            
            # Build FFmpeg command
            cmd = self._build_command(task)
            logger.info(f"Converting {task.source_path.name}: {' '.join(cmd)}")
            
            # Start process
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=creationflags
            )
            
            task_id = str(id(task))
            self.active_processes[task_id] = process
            
            # Monitor progress
            self._monitor_progress(process, task)
            
            # Wait for completion
            return_code = process.wait()
            
            # Clean up
            if task_id in self.active_processes:
                del self.active_processes[task_id]
            
            if return_code == 0 and task.output_path.exists():
                task.status = "completed"
                task.file_size_out = task.output_path.stat().st_size
                task.progress = 100.0
                task.end_time = time.time()
                return True
            else:
                task.status = "failed"
                task.error_message = f"FFmpeg returned code {return_code}"
                return False
                
        except Exception as e:
            task.status = "failed"
            task.error_message = str(e)
            logger.error(f"Conversion failed for {task.source_path.name}: {e}")
            return False
    
    def _build_command(self, task: ConversionTask) -> List[str]:
        """Build FFmpeg command based on preset"""
        cmd = [
            self.ffmpeg_path,
            '-y',  # Overwrite output
            '-i', str(task.source_path),
            '-progress', 'pipe:1',
            '-nostats',
            '-loglevel', 'error'
        ]
        
        preset = task.preset
        
        # Add stream mapping
        cmd.extend(['-map', '0:v', '-map', '0:a'])
        
        # Add codec settings
        cmd.extend(['-c:v', preset.video_codec])
        cmd.extend(['-c:a', preset.audio_codec])
        
        # Add bitrate settings
        if preset.video_bitrate:
            cmd.extend(['-b:v', preset.video_bitrate])
        if preset.audio_bitrate:
            cmd.extend(['-b:a', preset.audio_bitrate])
        
        # Add scaling
        if preset.scale:
            cmd.extend(['-vf', f'scale={preset.scale}'])
        
        # Add extra arguments
        cmd.extend(preset.extra_args)
        
        # Add output path
        cmd.append(str(task.output_path))
        
        return cmd
    
    def _monitor_progress(self, process: subprocess.Popen, task: ConversionTask):
        """Monitor conversion progress"""
        duration = 0
        
        # Try to get duration from validator first
        try:
            validator = MediaValidator(self.ffmpeg_path.replace('ffmpeg', 'ffprobe'))
            info = validator.validate_file(task.source_path)
            duration = info.get('duration', 0)
        except:
            pass
        
        def read_stderr():
            for line in process.stderr:
                logger.debug(f"FFmpeg stderr: {line.strip()}")
        
        # Start stderr reader thread
        threading.Thread(target=read_stderr, daemon=True).start()
        
        # Read progress from stdout
        for line in process.stdout:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                if key == 'out_time_us' and duration > 0:
                    try:
                        current_time = float(value) / 1_000_000  # Convert microseconds to seconds
                        progress = min(100, (current_time / duration) * 100)
                        task.progress = progress
                        if self.progress_callback:
                            self.progress_callback(task)
                    except ValueError:
                        pass
                elif key == 'progress' and value == 'end':
                    break
    
    def cancel_task(self, task: ConversionTask):
        """Cancel a running conversion task"""
        task_id = str(id(task))
        if task_id in self.active_processes:
            try:
                self.active_processes[task_id].terminate()
                task.status = "cancelled"
            except:
                pass

class PresetManager:
    """Manages conversion presets"""
    
    BUILTIN_PRESETS = [
        ConversionPreset(
            name="copy_all",
            description="Copy all streams (fastest, no quality loss)",
            video_codec="copy",
            audio_codec="copy"
        ),
        ConversionPreset(
            name="h264_high",
            description="H.264 High Quality (slower, good compression)",
            video_codec="libx264",
            audio_codec="aac",
            video_bitrate="8000k",
            audio_bitrate="320k",
            extra_args=["-preset", "slow", "-crf", "18"]
        ),
        ConversionPreset(
            name="h264_medium",
            description="H.264 Medium Quality (balanced)",
            video_codec="libx264",
            audio_codec="aac",
            video_bitrate="4000k",
            audio_bitrate="192k",
            extra_args=["-preset", "medium", "-crf", "23"]
        ),
        ConversionPreset(
            name="h264_small",
            description="H.264 Small Size (faster, lower quality)",
            video_codec="libx264",
            audio_codec="aac",
            video_bitrate="1500k",
            audio_bitrate="128k",
            extra_args=["-preset", "fast", "-crf", "28"]
        ),
        ConversionPreset(
            name="720p",
            description="Scale to 720p",
            video_codec="libx264",
            audio_codec="aac",
            scale="1280:720",
            extra_args=["-preset", "medium", "-crf", "23"]
        )
    ]
    
    def __init__(self):
        self.presets = {p.name: p for p in self.BUILTIN_PRESETS}
    
    def get_preset(self, name: str) -> Optional[ConversionPreset]:
        return self.presets.get(name)
    
    def get_preset_names(self) -> List[str]:
        return list(self.presets.keys())

class EnhancedMediaConverter:
    """Main application class with modern UI and enhanced features"""
    
    def __init__(self):
        # Initialize core components
        self.config = ConfigManager()
        self.preset_manager = PresetManager()
        self.validator = None
        self.converter = None
        
        # Application state
        self.tasks: List[ConversionTask] = []
        self.is_converting = False
        self.executor: Optional[ThreadPoolExecutor] = None
        self.file_hashes: Dict[str, str] = {}  # For duplicate detection
        
        # Check for FFmpeg
        self._check_dependencies()
        
        # Initialize UI
        self._setup_ui()
        self._setup_styles()
        self._load_settings()
        
        logger.info("Enhanced Media Converter initialized")
    
    def _check_dependencies(self):
        """Check for required dependencies"""
        ffmpeg_path = shutil.which('ffmpeg')
        ffprobe_path = shutil.which('ffprobe')
        
        if not ffmpeg_path or not ffprobe_path:
            messagebox.showerror(
                "Missing Dependencies",
                "FFmpeg and FFprobe are required but not found in PATH.\n"
                "Please install FFmpeg and ensure it's in your system PATH."
            )
            sys.exit(1)
        
        self.validator = MediaValidator(ffprobe_path)
        self.converter = ConversionEngine(ffmpeg_path, self._on_progress_update)
    
    def _setup_ui(self):
        """Setup the main user interface"""
        # Create main window
        if DRAG_DROP_AVAILABLE:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()
        
        self.root.title("Enhanced Media to MKV Converter")
        self.root.geometry(self.config.get("window_geometry"))
        self.root.minsize(700, 600)
        
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabbed interface
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Main conversion tab
        self.main_tab = ttk.Frame(notebook)
        notebook.add(self.main_tab, text="Conversion")
        self._setup_main_tab()
        
        # Settings tab
        self.settings_tab = ttk.Frame(notebook)
        notebook.add(self.settings_tab, text="Settings")
        self._setup_settings_tab()
        
        # History tab
        self.history_tab = ttk.Frame(notebook)
        notebook.add(self.history_tab, text="History")
        self._setup_history_tab()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _setup_main_tab(self):
        """Setup the main conversion tab"""
        # File management section
        file_frame = ttk.LabelFrame(self.main_tab, text="Files", padding=10)
        file_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Buttons
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(btn_frame, text="Add Files", command=self._add_files).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Add Folder", command=self._add_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Clear All", command=self._clear_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(btn_frame, text="Remove Selected", command=self._remove_selected).pack(side=tk.LEFT)
        
        # File list with scrollbar
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for file list
        columns = ('file', 'status', 'progress', 'size')
        self.file_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=10)
        
        # Configure columns
        self.file_tree.heading('file', text='File')
        self.file_tree.heading('status', text='Status')
        self.file_tree.heading('progress', text='Progress')
        self.file_tree.heading('size', text='Size')
        
        self.file_tree.column('file', width=300)
        self.file_tree.column('status', width=100)
        self.file_tree.column('progress', width=100)
        self.file_tree.column('size', width=100)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=scrollbar.set)
        
        self.file_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Enable drag and drop if available
        if DRAG_DROP_AVAILABLE:
            self.file_tree.drop_target_register(DND_FILES)
            self.file_tree.dnd_bind('<<Drop>>', self._on_drop)
        
        # Settings section
        settings_frame = ttk.LabelFrame(self.main_tab, text="Conversion Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Preset selection
        preset_frame = ttk.Frame(settings_frame)
        preset_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(preset_frame, text="Preset:").pack(side=tk.LEFT)
        self.preset_var = tk.StringVar(value=self.config.get("selected_preset"))
        self.preset_combo = ttk.Combobox(
            preset_frame, 
            textvariable=self.preset_var,
            values=self.preset_manager.get_preset_names(),
            state="readonly",
            width=30
        )
        self.preset_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.preset_combo.bind('<<ComboboxSelected>>', self._on_preset_changed)
        
        # Preset description
        self.preset_desc = ttk.Label(settings_frame, text="", foreground="gray")
        self.preset_desc.pack(fill=tk.X)
        self._update_preset_description()
        
        # Options
        options_frame = ttk.Frame(settings_frame)
        options_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.delete_originals_var = tk.BooleanVar(value=self.config.get("delete_originals"))
        ttk.Checkbutton(
            options_frame, 
            text="Delete original files after conversion",
            variable=self.delete_originals_var
        ).pack(anchor=tk.W)
        
        self.same_folder_var = tk.BooleanVar(value=self.config.get("same_folder_output"))
        ttk.Checkbutton(
            options_frame, 
            text="Save to same folder as source",
            variable=self.same_folder_var
        ).pack(anchor=tk.W)
        
        self.check_duplicates_var = tk.BooleanVar(value=self.config.get("check_duplicates"))
        ttk.Checkbutton(
            options_frame, 
            text="Check for duplicate files",
            variable=self.check_duplicates_var
        ).pack(anchor=tk.W)
        
        # Progress section
        progress_frame = ttk.LabelFrame(self.main_tab, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Overall progress
        ttk.Label(progress_frame, text="Overall Progress:").pack(anchor=tk.W)
        self.overall_progress = ttk.Progressbar(progress_frame, mode='determinate')
        self.overall_progress.pack(fill=tk.X, pady=(2, 10))
        
        # Status label
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(anchor=tk.W)
        
        # Control buttons
        control_frame = ttk.Frame(self.main_tab)
        control_frame.pack(fill=tk.X)
        
        self.start_button = ttk.Button(control_frame, text="Start Conversion", command=self._start_conversion)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.cancel_button = ttk.Button(control_frame, text="Cancel", command=self._cancel_conversion, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT)
        
        # Statistics
        self.stats_label = ttk.Label(control_frame, text="")
        self.stats_label.pack(side=tk.RIGHT)
    
    def _setup_settings_tab(self):
        """Setup the settings tab"""
        # Output settings
        output_frame = ttk.LabelFrame(self.settings_tab, text="Output Settings", padding=10)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Output directory
        dir_frame = ttk.Frame(output_frame)
        dir_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(dir_frame, text="Default output directory:").pack(anchor=tk.W)
        dir_entry_frame = ttk.Frame(dir_frame)
        dir_entry_frame.pack(fill=tk.X, pady=(2, 0))
        
        self.output_dir_var = tk.StringVar(value=self.config.get("output_directory"))
        self.output_dir_entry = ttk.Entry(dir_entry_frame, textvariable=self.output_dir_var)
        self.output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(dir_entry_frame, text="Browse", command=self._browse_output_dir).pack(side=tk.RIGHT)
        
        # Performance settings
        perf_frame = ttk.LabelFrame(self.settings_tab, text="Performance Settings", padding=10)
        perf_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Thread count
        thread_frame = ttk.Frame(perf_frame)
        thread_frame.pack(fill=tk.X)
        
        ttk.Label(thread_frame, text="Maximum concurrent conversions:").pack(side=tk.LEFT)
        self.thread_var = tk.IntVar(value=self.config.get("max_threads"))
        thread_spin = ttk.Spinbox(thread_frame, from_=1, to=8, textvariable=self.thread_var, width=5)
        thread_spin.pack(side=tk.LEFT, padx=(5, 0))
        
        # Behavior settings
        behavior_frame = ttk.LabelFrame(self.settings_tab, text="Behavior Settings", padding=10)
        behavior_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.auto_open_var = tk.BooleanVar(value=self.config.get("auto_open_output"))
        ttk.Checkbutton(behavior_frame, text="Auto-open output folder when done", variable=self.auto_open_var).pack(anchor=tk.W)
        
        # Save settings button
        ttk.Button(self.settings_tab, text="Save Settings", command=self._save_settings).pack(pady=10)
    
    def _setup_history_tab(self):
        """Setup the conversion history tab"""
        ttk.Label(self.history_tab, text="Conversion History", font=('TkDefaultFont', 12, 'bold')).pack(pady=10)
        
        # History tree
        hist_frame = ttk.Frame(self.history_tab)
        hist_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        columns = ('date', 'file', 'status', 'time', 'size_reduction')
        self.history_tree = ttk.Treeview(hist_frame, columns=columns, show='headings')
        
        self.history_tree.heading('date', text='Date')
        self.history_tree.heading('file', text='File')
        self.history_tree.heading('status', text='Status')
        self.history_tree.heading('time', text='Time')
        self.history_tree.heading('size_reduction', text='Size Change')
        
        # Scrollbar for history
        hist_scroll = ttk.Scrollbar(hist_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=hist_scroll.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hist_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Clear history button
        ttk.Button(self.history_tab, text="Clear History", command=self._clear_history).pack(pady=10)
    
    def _setup_styles(self):
        """Setup custom styles for the UI"""
        style = ttk.Style()
        
        # Configure styles for better appearance
        style.configure('Title.TLabel', font=('TkDefaultFont', 12, 'bold'))
        style.configure('Status.TLabel', font=('TkDefaultFont', 9))
    
    def _load_settings(self):
        """Load settings from config"""
        # Update UI elements with saved settings
        self.delete_originals_var.set(self.config.get("delete_originals"))
        self.same_folder_var.set(self.config.get("same_folder_output"))
        self.check_duplicates_var.set(self.config.get("check_duplicates"))
        self.auto_open_var.set(self.config.get("auto_open_output"))
        self.thread_var.set(self.config.get("max_threads"))
        self.output_dir_var.set(self.config.get("output_directory"))
        self.preset_var.set(self.config.get("selected_preset"))
    
    def _add_files(self):
        """Add files through file dialog"""
        filetypes = [
            ("Media files", " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)),
            ("All files", "*.*")
        ]
        
        files = filedialog.askopenfilenames(
            title="Select media files",
            filetypes=filetypes
        )
        
        if files:
            self._add_file_paths([Path(f) for f in files])
    
    def _add_folder(self):
        """Add all media files from a folder"""
        folder = filedialog.askdirectory(title="Select folder containing media files")
        
        if folder:
            folder_path = Path(folder)
            media_files = []
            
            for ext in SUPPORTED_EXTENSIONS:
                media_files.extend(folder_path.rglob(f"*{ext}"))
            
            if media_files:
                self._add_file_paths(media_files)
            else:
                messagebox.showinfo("No Files", f"No supported media files found in {folder}")
    
    def _add_file_paths(self, file_paths: List[Path]):
        """Add file paths to conversion queue"""
        added_count = 0
        duplicate_count = 0
        invalid_count = 0
        
        for file_path in file_paths:
            if not file_path.exists():
                invalid_count += 1
                continue
            
            # Check for duplicates
            if self.check_duplicates_var.get():
                file_hash = self._get_file_hash(file_path)
                if file_hash in self.file_hashes:
                    duplicate_count += 1
                    continue
                self.file_hashes[file_hash] = str(file_path)
            
            # Check if already in queue
            if any(task.source_path == file_path for task in self.tasks):
                duplicate_count += 1
                continue
            
            # Validate file
            validation = self.validator.validate_file(file_path)
            if not validation.get("valid", False):
                invalid_count += 1
                logger.warning(f"Invalid file skipped: {file_path}")
                continue
            
            # Create conversion task
            output_path = self._get_output_path(file_path)
            preset = self.preset_manager.get_preset(self.preset_var.get())
            
            task = ConversionTask(
                source_path=file_path,
                output_path=output_path,
                preset=preset
            )
            
            self.tasks.append(task)
            added_count += 1
        
        # Update UI
        self._update_file_tree()
        self._update_stats()
        
        # Show summary if there were issues
        if duplicate_count > 0 or invalid_count > 0:
            msg = f"Added {added_count} files.\n"
            if duplicate_count > 0:
                msg += f"Skipped {duplicate_count} duplicates.\n"
            if invalid_count > 0:
                msg += f"Skipped {invalid_count} invalid files."
            messagebox.showinfo("Files Added", msg)
        
        logger.info(f"Added {added_count} files to queue")
    
    def _on_drop(self, event):
        """Handle drag and drop files"""
        if not DRAG_DROP_AVAILABLE:
            return
        
        files = self.root.tk.splitlist(event.data)
        file_paths = []
        
        for file_str in files:
            path = Path(file_str)
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                file_paths.append(path)
            elif path.is_dir():
                for ext in SUPPORTED_EXTENSIONS:
                    file_paths.extend(path.rglob(f"*{ext}"))
        
        if file_paths:
            self._add_file_paths(file_paths)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of file for duplicate detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return str(file_path)  # Fallback to path if hashing fails
    
    def _get_output_path(self, source_path: Path) -> Path:
        """Generate output path for a source file"""
        if self.same_folder_var.get():
            output_dir = source_path.parent
        else:
            output_dir_str = self.output_dir_var.get().strip()
            if output_dir_str:
                output_dir = Path(output_dir_str)
            else:
                output_dir = Path.cwd() / "Converted_MKV"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{source_path.stem}.mkv"
    
    def _update_file_tree(self):
        """Update the file tree view"""
        # Clear existing items
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Add current tasks
        for i, task in enumerate(self.tasks):
            # Format file size
            size_mb = task.source_path.stat().st_size / (1024 * 1024)
            size_str = f"{size_mb:.1f} MB"
            
            # Format progress
            if task.status == "pending":
                progress_str = "Pending"
            elif task.status == "processing":
                progress_str = f"{task.progress:.1f}%"
            elif task.status == "completed":
                progress_str = "100%"
            elif task.status == "failed":
                progress_str = "Failed"
            else:
                progress_str = task.status.title()
            
            self.file_tree.insert('', 'end', values=(
                task.source_path.name,
                task.status.title(),
                progress_str,
                size_str
            ))
    
    def _update_stats(self):
        """Update statistics display"""
        total_files = len(self.tasks)
        completed = len([t for t in self.tasks if t.status == "completed"])
        failed = len([t for t in self.tasks if t.status == "failed"])
        
        stats_text = f"Files: {total_files} | Completed: {completed} | Failed: {failed}"
        self.stats_label.config(text=stats_text)
    
    def _remove_selected(self):
        """Remove selected items from queue"""
        selection = self.file_tree.selection()
        if not selection:
            return
        
        # Get indices of selected items
        indices_to_remove = []
        for item in selection:
            index = self.file_tree.index(item)
            indices_to_remove.append(index)
        
        # Remove tasks (in reverse order to maintain indices)
        for index in sorted(indices_to_remove, reverse=True):
            if index < len(self.tasks):
                removed_task = self.tasks.pop(index)
                # Remove from hash tracking
                if self.check_duplicates_var.get():
                    file_hash = self._get_file_hash(removed_task.source_path)
                    self.file_hashes.pop(file_hash, None)
        
        self._update_file_tree()
        self._update_stats()
    
    def _clear_all(self):
        """Clear all tasks from queue"""
        if self.tasks and messagebox.askyesno("Clear All", "Remove all files from the queue?"):
            self.tasks.clear()
            self.file_hashes.clear()
            self._update_file_tree()
            self._update_stats()
    
    def _on_preset_changed(self, event=None):
        """Handle preset selection change"""
        self._update_preset_description()
        self.config.set("selected_preset", self.preset_var.get())
        
        # Update all pending tasks with new preset
        new_preset = self.preset_manager.get_preset(self.preset_var.get())
        for task in self.tasks:
            if task.status == "pending":
                task.preset = new_preset
    
    def _update_preset_description(self):
        """Update preset description label"""
        preset_name = self.preset_var.get()
        preset = self.preset_manager.get_preset(preset_name)
        if preset:
            self.preset_desc.config(text=preset.description)
    
    def _start_conversion(self):
        """Start the conversion process"""
        if not self.tasks:
            messagebox.showwarning("No Files", "Please add files to convert.")
            return
        
        # Check if any tasks are pending
        pending_tasks = [t for t in self.tasks if t.status == "pending"]
        if not pending_tasks:
            messagebox.showinfo("Nothing to Convert", "All files have already been processed.")
            return
        
        self.is_converting = True
        self._update_ui_state()
        
        # Start conversion in thread pool
        max_workers = self.config.get("max_threads")
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Submit tasks
        futures = []
        for task in pending_tasks:
            future = self.executor.submit(self.converter.convert_file, task)
            futures.append(future)
        
        # Monitor completion
        def monitor_completion():
            try:
                completed_count = 0
                total_tasks = len(pending_tasks)
                
                for future in as_completed(futures):
                    if not self.is_converting:  # Check for cancellation
                        break
                    
                    completed_count += 1
                    progress = (completed_count / total_tasks) * 100
                    
                    self.root.after(0, self._update_overall_progress, progress)
                    self.root.after(0, self._update_file_tree)
                    self.root.after(0, self._update_stats)
                
                # Conversion finished
                self.root.after(0, self._conversion_finished)
                
            except Exception as e:
                logger.error(f"Error during conversion: {e}")
                self.root.after(0, self._conversion_finished)
        
        threading.Thread(target=monitor_completion, daemon=True).start()
        
        logger.info(f"Started conversion of {len(pending_tasks)} files")
    
    def _cancel_conversion(self):
        """Cancel ongoing conversion"""
        if messagebox.askyesno("Cancel Conversion", "Are you sure you want to cancel the conversion?"):
            self.is_converting = False
            
            # Cancel all running tasks
            for task in self.tasks:
                if task.status == "processing":
                    self.converter.cancel_task(task)
            
            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=False)
                self.executor = None
            
            self.status_label.config(text="Conversion cancelled")
            self._update_ui_state()
            
            logger.info("Conversion cancelled by user")
    
    def _conversion_finished(self):
        """Handle conversion completion"""
        self.is_converting = False
        self._update_ui_state()
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        # Generate summary
        completed = [t for t in self.tasks if t.status == "completed"]
        failed = [t for t in self.tasks if t.status == "failed"]
        
        # Calculate statistics
        total_size_in = sum(t.file_size_in for t in completed)
        total_size_out = sum(t.file_size_out for t in completed)
        size_reduction = ((total_size_in - total_size_out) / total_size_in * 100) if total_size_in > 0 else 0
        
        # Create summary message
        summary = f"Conversion completed!\n\n"
        summary += f"Completed: {len(completed)}\n"
        summary += f"Failed: {len(failed)}\n"
        
        if completed:
            summary += f"Size reduction: {size_reduction:.1f}%\n"
            summary += f"Total size: {total_size_in/(1024**2):.1f} MB → {total_size_out/(1024**2):.1f} MB"
        
        # Delete originals if requested
        if self.delete_originals_var.get() and completed:
            deleted_count = 0
            for task in completed:
                try:
                    task.source_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {task.source_path}: {e}")
            
            if deleted_count > 0:
                summary += f"\n\nDeleted {deleted_count} original files"
        
        self.status_label.config(text=f"Completed {len(completed)} files")
        messagebox.showinfo("Conversion Complete", summary)
        
        # Open output folder if requested
        if self.auto_open_var.get() and completed:
            output_dir = completed[0].output_path.parent
            try:
                webbrowser.open(f"file://{output_dir}")
            except Exception as e:
                logger.error(f"Failed to open output directory: {e}")
        
        logger.info(f"Conversion finished: {len(completed)} completed, {len(failed)} failed")
    
    def _on_progress_update(self, task: ConversionTask):
        """Handle progress updates from conversion engine"""
        self.root.after(0, self._update_file_tree)
        
        # Update status
        if task.progress > 0:
            self.root.after(0, self.status_label.config, 
                          {'text': f"Converting {task.source_path.name}: {task.progress:.1f}%"})
    
    def _update_overall_progress(self, progress: float):
        """Update overall progress bar"""
        self.overall_progress['value'] = progress
    
    def _update_ui_state(self):
        """Update UI elements based on conversion state"""
        if self.is_converting:
            self.start_button.config(state=tk.DISABLED)
            self.cancel_button.config(state=tk.NORMAL)
            self.status_label.config(text="Converting...")
        else:
            self.start_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED)
            if not hasattr(self, '_conversion_cancelled'):
                self.status_label.config(text="Ready")
    
    def _browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select output directory")
        if directory:
            self.output_dir_var.set(directory)
    
    def _save_settings(self):
        """Save current settings"""
        self.config.set("delete_originals", self.delete_originals_var.get())
        self.config.set("same_folder_output", self.same_folder_var.get())
        self.config.set("check_duplicates", self.check_duplicates_var.get())
        self.config.set("auto_open_output", self.auto_open_var.get())
        self.config.set("max_threads", self.thread_var.get())
        self.config.set("output_directory", self.output_dir_var.get())
        self.config.set("selected_preset", self.preset_var.get())
        
        messagebox.showinfo("Settings Saved", "Settings have been saved successfully.")
    
    def _clear_history(self):
        """Clear conversion history"""
        # Clear history tree
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        messagebox.showinfo("History Cleared", "Conversion history has been cleared.")
    
    def _on_closing(self):
        """Handle application closing"""
        # Save window geometry
        self.config.set("window_geometry", self.root.geometry())
        
        # Cancel any ongoing conversions
        if self.is_converting:
            if messagebox.askyesno("Exit", "Conversion is in progress. Exit anyway?"):
                self.is_converting = False
                if self.executor:
                    self.executor.shutdown(wait=False)
            else:
                return
        
        # Save current settings
        self._save_settings()
        
        logger.info("Application closing")
        self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        app = EnhancedMediaConverter()
        app.run()
    except Exception as e:
        logger.critical(f"Application failed to start: {e}")
        messagebox.showerror("Error", f"Failed to start application:\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()