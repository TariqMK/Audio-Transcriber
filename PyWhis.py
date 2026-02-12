import sys
import os
import json
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta
import whisper
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QProgressBar, 
                             QTextEdit, QFileDialog, QListWidget, QComboBox,
                             QGroupBox, QCheckBox, QLineEdit, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor


# Model download URLs
MODEL_URLS = {
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v2": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
}

MODEL_SIZES = {
    "tiny": "~75 MB",
    "base": "~142 MB",
    "small": "~466 MB",
    "medium": "~1.5 GB",
    "large-v2": "~3 GB",
    "large-v3": "~3 GB"
}


class ModelDownloadWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished_download = pyqtSignal(str)  # model_path
    error = pyqtSignal(str)
    
    def __init__(self, model_name, models_dir):
        super().__init__()
        self.model_name = model_name
        self.models_dir = models_dir
        
    def run(self):
        try:
            url = MODEL_URLS[self.model_name]
            model_path = Path(self.models_dir) / f"{self.model_name}.pt"
            
            # Check if already exists
            if model_path.exists():
                self.status.emit(f"Model already exists, loading...")
                self.finished_download.emit(str(model_path))
                return
            
            # Ensure directory exists
            Path(self.models_dir).mkdir(parents=True, exist_ok=True)
            
            # Download with progress
            self.status.emit(f"Downloading {self.model_name} model ({MODEL_SIZES[self.model_name]})...")
            
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = int((downloaded / total_size) * 100)
                    self.progress.emit(percent)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    self.status.emit(f"Downloading: {mb_downloaded:.1f} MB / {mb_total:.1f} MB")
            
            urllib.request.urlretrieve(url, str(model_path), reporthook=report_progress)
            
            self.status.emit("Download complete, loading model...")
            self.finished_download.emit(str(model_path))
            
        except Exception as e:
            self.error.emit(f"Download failed: {str(e)}")


class TranscriptionWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    file_complete = pyqtSignal(str, float, int)
    all_complete = pyqtSignal(float, int, int)
    error = pyqtSignal(str, str)  # filename, error_message
    
    def __init__(self, files, model, output_dir, error_log_path):
        super().__init__()
        self.files = files
        self.model = model
        self.output_dir = output_dir
        self.error_log_path = error_log_path
        self.is_cancelled = False
        self.errors = []
        
    def run(self):
        total_files = len(self.files)
        total_words = 0
        successful_files = 0
        start_time = self.get_time()
        
        for idx, file_path in enumerate(self.files):
            if self.is_cancelled:
                self.status.emit("Transcription cancelled")
                break
                
            file_name = Path(file_path).name
            self.status.emit(f"Transcribing {file_name} ({idx + 1}/{total_files})...")
            
            try:
                # Transcribe
                file_start = self.get_time()
                result = self.model.transcribe(file_path, verbose=False)
                file_duration = self.get_time() - file_start
                
                # Count words
                word_count = len(result['text'].split())
                total_words += word_count
                
                # Save output - create output directory if needed
                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
                output_filename = Path(file_path).stem + '.txt'
                output_path = Path(self.output_dir) / output_filename
                self.save_transcription(result, output_path)
                
                successful_files += 1
                self.file_complete.emit(file_name, file_duration, word_count)
                
            except Exception as e:
                error_msg = f"Error transcribing {file_name}: {str(e)}"
                self.errors.append((file_name, str(e)))
                self.error.emit(file_name, str(e))
                self.status.emit(f"⚠️ Skipped {file_name} due to error")
            
            # Update progress
            progress_percent = int(((idx + 1) / total_files) * 100)
            self.progress.emit(progress_percent)
        
        # Write error log if there were errors
        if self.errors and self.error_log_path:
            self.write_error_log()
        
        total_duration = self.get_time() - start_time
        self.all_complete.emit(total_duration, total_words, successful_files)
    
    def save_transcription(self, result, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
    
    def write_error_log(self):
        try:
            # Ensure log directory exists
            log_dir = Path(self.error_log_path)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"transcription_errors_{timestamp}.log"
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"Transcription Error Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                for filename, error in self.errors:
                    f.write(f"File: {filename}\n")
                    f.write(f"Error: {error}\n")
                    f.write("-" * 70 + "\n\n")
        except Exception as e:
            self.error.emit("Error Log", f"Could not write error log: {str(e)}")
    
    def get_time(self):
        import time
        return time.time()
    
    def cancel(self):
        self.is_cancelled = True


class WhisperGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.files = []
        self.worker = None
        self.download_worker = None
        self.model = None
        self.model_loaded = False
        self.models_dir = ""
        self.error_log_path = ""
        self.output_dir = ""
        self.settings_file = Path.home() / ".whisper_gui_settings.json"
        
        self.load_settings()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Whisper Transcription Tool")
        self.setGeometry(100, 100, 950, 800)
        
        # Set dark theme
        self.set_dark_theme()
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("🎙️ Whisper Transcription Tool")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        
        # Models Directory Configuration
        models_dir_group = QGroupBox("Models Directory")
        models_dir_layout = QHBoxLayout()
        
        models_dir_layout.addWidget(QLabel("Models Folder:"))
        self.models_dir_input = QLineEdit()
        self.models_dir_input.setText(self.models_dir)
        self.models_dir_input.setReadOnly(True)
        self.models_dir_input.setPlaceholderText("Click 'Browse' to select models directory")
        models_dir_layout.addWidget(self.models_dir_input)
        
        self.browse_models_dir_btn = QPushButton("Browse")
        self.browse_models_dir_btn.clicked.connect(self.select_models_dir)
        models_dir_layout.addWidget(self.browse_models_dir_btn)
        
        models_dir_group.setLayout(models_dir_layout)
        main_layout.addWidget(models_dir_group)
        
        # Model Selection & Download
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        for model_name in MODEL_URLS.keys():
            self.model_combo.addItem(f"{model_name} ({MODEL_SIZES[model_name]})", model_name)
        self.model_combo.setCurrentIndex(3)  # Default to medium
        model_select_layout.addWidget(self.model_combo)
        
        self.load_model_btn = QPushButton("Download & Load Model")
        self.load_model_btn.setStyleSheet("background-color: #2196F3; font-weight: bold; padding: 10px;")
        self.load_model_btn.clicked.connect(self.download_and_load_model)
        model_select_layout.addWidget(self.load_model_btn)
        
        model_select_layout.addStretch()
        model_layout.addLayout(model_select_layout)
        
        # Model download progress
        self.download_progress = QProgressBar()
        self.download_progress.setValue(0)
        self.download_progress.setVisible(False)
        model_layout.addWidget(self.download_progress)
        
        # Model status
        self.model_status_label = QLabel("⚪ Model not loaded")
        self.model_status_label.setStyleSheet("color: #ff9800; font-weight: bold;")
        model_layout.addWidget(self.model_status_label)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # Settings Group
        settings_group = QGroupBox("Output Settings")
        settings_layout = QVBoxLayout()
        
        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_input = QLineEdit()
        self.output_dir_input.setText(self.output_dir)
        self.output_dir_input.setReadOnly(True)
        self.output_dir_input.setPlaceholderText("Click 'Browse' to select output directory")
        output_dir_layout.addWidget(self.output_dir_input)
        
        self.browse_output_dir_btn = QPushButton("Browse")
        self.browse_output_dir_btn.clicked.connect(self.select_output_dir)
        output_dir_layout.addWidget(self.browse_output_dir_btn)
        settings_layout.addLayout(output_dir_layout)
        
        error_log_layout = QHBoxLayout()
        error_log_layout.addWidget(QLabel("Error Log Directory:"))
        self.error_log_input = QLineEdit()
        self.error_log_input.setText(self.error_log_path)
        self.error_log_input.setReadOnly(True)
        self.error_log_input.setPlaceholderText("Click 'Browse' to select error log directory")
        error_log_layout.addWidget(self.error_log_input)
        
        self.browse_error_log_btn = QPushButton("Browse")
        self.browse_error_log_btn.clicked.connect(self.select_error_log_dir)
        error_log_layout.addWidget(self.browse_error_log_btn)
        settings_layout.addLayout(error_log_layout)
        
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # File selection
        file_layout = QHBoxLayout()
        self.select_file_btn = QPushButton("Select File(s)")
        self.select_file_btn.clicked.connect(self.select_files)
        file_layout.addWidget(self.select_file_btn)
        
        self.clear_btn = QPushButton("Clear List")
        self.clear_btn.clicked.connect(self.clear_files)
        file_layout.addWidget(self.clear_btn)
        
        file_layout.addStretch()
        main_layout.addLayout(file_layout)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(150)
        main_layout.addWidget(QLabel("Selected Files:"))
        main_layout.addWidget(self.file_list)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # Status
        self.status_label = QLabel("Ready - Please set models directory and load a model")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #ff9800; font-size: 14px; font-weight: bold;")
        main_layout.addWidget(self.status_label)
        
        # Stats display
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        self.stats_text.setPlaceholderText("Statistics will appear here after transcription...")
        stats_layout.addWidget(self.stats_text)
        stats_group.setLayout(stats_layout)
        main_layout.addWidget(stats_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Transcription")
        self.start_btn.setStyleSheet("background-color: #4CAF50; font-size: 14px; font-weight: bold; padding: 10px;")
        self.start_btn.setEnabled(False)  # Disabled until model is loaded
        self.start_btn.clicked.connect(self.start_transcription)
        button_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("background-color: #f44336; font-size: 14px; font-weight: bold; padding: 10px;")
        self.cancel_btn.clicked.connect(self.cancel_transcription)
        button_layout.addWidget(self.cancel_btn)
        
        main_layout.addLayout(button_layout)
        
    def set_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        
        self.setPalette(palette)
    
    def load_settings(self):
        """Load saved settings from file"""
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.models_dir = settings.get('models_dir', '')
                    self.error_log_path = settings.get('error_log_path', '')
                    self.output_dir = settings.get('output_dir', '')
        except Exception as e:
            print(f"Could not load settings: {e}")
            self.models_dir = ""
            self.error_log_path = ""
            self.output_dir = ""
    
    def save_settings(self):
        """Save settings to file"""
        try:
            settings = {
                'models_dir': self.models_dir,
                'error_log_path': self.error_log_path,
                'output_dir': self.output_dir
            }
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Could not save settings: {e}")
    
    def select_models_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Models Directory")
        if directory:
            self.models_dir = directory
            self.models_dir_input.setText(directory)
            self.save_settings()
            self.status_label.setText("Models directory set - Select a model and click 'Download & Load'")
            self.status_label.setStyleSheet("color: #2196F3; font-size: 14px; font-weight: bold;")
    
    def download_and_load_model(self):
        if not self.models_dir:
            QMessageBox.warning(self, "Warning", "Please select a models directory first!")
            return
        
        model_name = self.model_combo.currentData()
        
        # Disable controls
        self.load_model_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.browse_models_dir_btn.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)
        
        # Start download worker
        self.download_worker = ModelDownloadWorker(model_name, self.models_dir)
        self.download_worker.progress.connect(self.update_download_progress)
        self.download_worker.status.connect(self.update_download_status)
        self.download_worker.finished_download.connect(self.load_downloaded_model)
        self.download_worker.error.connect(self.download_error)
        self.download_worker.start()
    
    def update_download_progress(self, value):
        self.download_progress.setValue(value)
    
    def update_download_status(self, message):
        self.model_status_label.setText(message)
        self.model_status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
    
    def load_downloaded_model(self, model_path):
        try:
            self.model_status_label.setText("⏳ Loading model into memory...")
            QApplication.processEvents()
            
            # Load the model
            self.model = whisper.load_model(model_path)
            self.model_loaded = True
            
            model_name = Path(model_path).stem
            self.model_status_label.setText(f"✅ Model '{model_name}' loaded and ready")
            self.model_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.status_label.setText("Ready to transcribe")
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 14px; font-weight: bold;")
            self.start_btn.setEnabled(True)
            
        except Exception as e:
            self.model_status_label.setText(f"❌ Model load failed")
            self.model_status_label.setStyleSheet("color: #f44336; font-weight: bold;")
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
        
        finally:
            # Re-enable controls
            self.load_model_btn.setEnabled(True)
            self.model_combo.setEnabled(True)
            self.browse_models_dir_btn.setEnabled(True)
            self.download_progress.setVisible(False)
    
    def download_error(self, message):
        self.model_status_label.setText(f"❌ {message}")
        self.model_status_label.setStyleSheet("color: #f44336; font-weight: bold;")
        QMessageBox.critical(self, "Download Error", message)
        
        # Re-enable controls
        self.load_model_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.browse_models_dir_btn.setEnabled(True)
        self.download_progress.setVisible(False)
    
    def select_error_log_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Error Log Directory")
        if directory:
            self.error_log_path = directory
            self.error_log_input.setText(directory)
            self.save_settings()
    
    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory for Transcriptions")
        if directory:
            self.output_dir = directory
            self.output_dir_input.setText(directory)
            self.save_settings()
    
    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files",
            "",
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg *.mp4 *.avi *.mkv);;All Files (*)"
        )
        if files:
            self.files.extend(files)
            self.file_list.clear()
            self.file_list.addItems([Path(f).name for f in self.files])
            
    def clear_files(self):
        self.files = []
        self.file_list.clear()
        self.stats_text.clear()
        
    def start_transcription(self):
        if not self.model_loaded:
            QMessageBox.warning(self, "Warning", "Please load a model first!")
            return
            
        if not self.files:
            self.status_label.setText("⚠️ Please select at least one file")
            self.status_label.setStyleSheet("color: #ff9800; font-size: 14px; font-weight: bold;")
            return
        
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Please select an output directory for transcriptions!")
            return
        
        if not self.error_log_path:
            reply = QMessageBox.question(
                self,
                "No Error Log Directory",
                "You haven't set an error log directory. Errors will not be saved to a log file.\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.select_file_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.load_model_btn.setEnabled(False)
        self.browse_models_dir_btn.setEnabled(False)
        self.browse_error_log_btn.setEnabled(False)
        self.browse_output_dir_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.progress_bar.setValue(0)
        self.stats_text.clear()
        
        self.worker = TranscriptionWorker(
            self.files, 
            self.model, 
            self.output_dir,
            self.error_log_path
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.update_status)
        self.worker.file_complete.connect(self.file_completed)
        self.worker.all_complete.connect(self.all_completed)
        self.worker.error.connect(self.show_error)
        self.worker.finished.connect(self.transcription_finished)
        self.worker.start()
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def update_status(self, message):
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #2196F3; font-size: 14px; font-weight: bold;")
        
    def file_completed(self, filename, duration, word_count):
        stats = self.stats_text.toPlainText()
        stats += f"✓ {filename}: {duration:.2f}s, {word_count} words\n"
        self.stats_text.setPlainText(stats)
        
    def show_error(self, filename, error):
        stats = self.stats_text.toPlainText()
        stats += f"✗ {filename}: ERROR - {error}\n"
        self.stats_text.setPlainText(stats)
        
    def all_completed(self, total_time, total_words, successful_files):
        error_count = len(self.files) - successful_files
        
        if error_count == 0:
            self.status_label.setText("✅ All transcriptions complete!")
            self.status_label.setStyleSheet("color: #4CAF50; font-size: 14px; font-weight: bold;")
        else:
            self.status_label.setText(f"⚠️ Completed with {error_count} error(s)")
            self.status_label.setStyleSheet("color: #ff9800; font-size: 14px; font-weight: bold;")
        
        stats = self.stats_text.toPlainText()
        stats += f"\n{'='*50}\n"
        stats += f"Total time: {total_time:.2f}s\n"
        stats += f"Total words: {total_words}\n"
        stats += f"Files processed: {successful_files}/{len(self.files)}\n"
        if successful_files > 0:
            stats += f"Average: {total_time/successful_files:.2f}s per file\n"
        if error_count > 0:
            stats += f"\n⚠️ {error_count} file(s) failed - check error log in:\n{self.error_log_path}\n"
        self.stats_text.setPlainText(stats)
        
    def transcription_finished(self):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.select_file_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        self.browse_models_dir_btn.setEnabled(True)
        self.browse_error_log_btn.setEnabled(True)
        self.browse_output_dir_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        
    def cancel_transcription(self):
        if self.worker:
            self.worker.cancel()
            self.status_label.setText("Cancelling...")
            self.status_label.setStyleSheet("color: #ff9800; font-size: 14px; font-weight: bold;")


def main():
    app = QApplication(sys.argv)
    window = WhisperGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()