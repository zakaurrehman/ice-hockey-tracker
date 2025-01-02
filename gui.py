"""GUI application for hockey player tracking"""
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QProgressBar,
    QTextEdit, QSpinBox, QCheckBox, QGroupBox, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from src.enhanced_tracker import EnhancedPlayerTracker
from src.video_analyzer import VideoAnalyzer

class ProcessingThread(QThread):
    progress_update = pyqtSignal(int)
    analysis_update = pyqtSignal(str)
    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, video_path, jersey_number, jersey_color, output_dir, analyze_only=False):
        super().__init__()
        self.video_path = video_path
        self.jersey_number = jersey_number
        self.jersey_color = jersey_color
        self.output_dir = output_dir
        self.analyze_only = analyze_only

    def run(self):
        try:
            if self.analyze_only:
                self._run_analysis()
            else:
                self._run_tracking()
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _run_tracking(self):
        """Run player tracking"""
        try:
            tracker = EnhancedPlayerTracker(self.jersey_number, self.jersey_color)
            output_path = tracker.track_player(self.video_path, self.output_dir, self.progress_update)
            
            if output_path:
                self.finished.emit(output_path)
                
                # Get and display player summary
                stats = tracker.get_player_stats()
                self._display_stats(stats)
            else:
                self.error_occurred.emit("Failed to create highlight video")
                
        except Exception as e:
            self.error_occurred.emit(f"Tracking error: {str(e)}")
        finally:
            if 'tracker' in locals():
                tracker.cleanup()

    def _run_analysis(self):
        """Run video analysis"""
        try:
            analyzer = VideoAnalyzer(self.video_path)
            self.analysis_update.emit("Starting video analysis...")
            
            segments = analyzer.analyze_video(self.jersey_number, self.jersey_color)
            self.analysis_update.emit("\nAnalysis complete!")
            
            # Display segment information
            self._display_segments(segments)
            
            # Extract highlights
            highlight_paths = analyzer.extract_highlights(self.output_dir, 
                                                       self.jersey_number,
                                                       self.jersey_color)
            
            if highlight_paths:
                self.finished.emit(highlight_paths[0])  # Send first highlight path
            else:
                self.error_occurred.emit("No highlights created")
                
        except Exception as e:
            self.error_occurred.emit(f"Analysis error: {str(e)}")
        finally:
            if 'analyzer' in locals():
                analyzer.cleanup()

    def _display_stats(self, stats):
        """Display player statistics"""
        self.analysis_update.emit("\nPlayer Statistics:")
        for player, stat in stats.items():
            self.analysis_update.emit(
                f"\nPlayer: {player}\n"
                f"Total Frames: {stat['total_frames']}\n"
                f"Segments: {stat['segments']}"
            )

    def _display_segments(self, segments):
        """Display segment information"""
        self.analysis_update.emit("\nSegment Information:")
        for i, segment in enumerate(segments):
            self.analysis_update.emit(
                f"\nSegment {i+1}:"
                f"\nDuration: {segment.get_duration()} frames"
                f"\nPlayers: {', '.join(segment.players)}"
            )

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle('Hockey Player Tracker')
        self.setGeometry(100, 100, 800, 600)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # Add file selection
        file_group = self._create_file_selection()
        layout.addWidget(file_group)

        # Add player settings
        player_group = self._create_player_settings()
        layout.addWidget(player_group)

        # Add processing options
        options_group = self._create_processing_options()
        layout.addWidget(options_group)

        # Add progress tracking
        progress_group = self._create_progress_tracking()
        layout.addWidget(progress_group)

        # Add control buttons
        button_layout = self._create_control_buttons()
        layout.addLayout(button_layout)

        main_widget.setLayout(layout)

    def _create_file_selection(self):
        """Create file selection group"""
        group = QGroupBox("Video Selection")
        layout = QHBoxLayout()

        self.file_label = QLabel('No file selected')
        file_button = QPushButton('Select Video')
        file_button.clicked.connect(self.select_file)

        layout.addWidget(file_button)
        layout.addWidget(self.file_label)
        group.setLayout(layout)

        return group

    def _create_player_settings(self):
        """Create player settings group"""
        group = QGroupBox("Player Settings")
        layout = QVBoxLayout()

        # Jersey number selection
        number_layout = QHBoxLayout()
        number_layout.addWidget(QLabel('Jersey Number:'))
        self.number_spinner = QSpinBox()
        self.number_spinner.setRange(0, 99)
        number_layout.addWidget(self.number_spinner)
        layout.addLayout(number_layout)

        # Jersey color selection
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel('Jersey Color:'))
        self.color_combo = QComboBox()
        self.color_combo.addItems([
            'black-jersey', 'white-red-jersey', 'white-jersey',
            'black-white-numbers', 'game-red', 'game-blue',
            'game-black', 'black-light-blue', 'white-red-blue'
        ])
        color_layout.addWidget(self.color_combo)
        layout.addLayout(color_layout)

        group.setLayout(layout)
        return group

    def _create_processing_options(self):
        """Create processing options group"""
        group = QGroupBox("Processing Options")
        layout = QVBoxLayout()

        self.analyze_only_check = QCheckBox("Analysis Only")
        layout.addWidget(self.analyze_only_check)

        self.save_frames_check = QCheckBox("Save Debug Frames")
        layout.addWidget(self.save_frames_check)

        group.setLayout(layout)
        return group

    def _create_progress_tracking(self):
        """Create progress tracking group"""
        group = QGroupBox("Progress")
        layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        group.setLayout(layout)
        return group

    def _create_control_buttons(self):
        """Create control buttons"""
        layout = QHBoxLayout()

        self.start_button = QPushButton('Start Processing')
        self.start_button.clicked.connect(self.start_processing)
        layout.addWidget(self.start_button)

        self.clear_button = QPushButton('Clear Output')
        self.clear_button.clicked.connect(self.clear_output)
        layout.addWidget(self.clear_button)

        return layout

    def select_file(self):
        """Handle file selection"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            'Select Video File',
            '',
            'Video Files (*.mp4 *.avi);;All Files (*)'
        )
        if file_name:
            self.file_label.setText(file_name)

    def start_processing(self):
        """Start video processing"""
        if self.file_label.text() == 'No file selected':
            QMessageBox.warning(self, 'Error', 'Please select a video file first.')
            return

        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.output_text.clear()

        # Create output directory
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)

        # Start processing thread
        self.processing_thread = ProcessingThread(
            self.file_label.text(),
            self.number_spinner.value(),
            self.color_combo.currentText(),
            output_dir,
            self.analyze_only_check.isChecked()
        )
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.analysis_update.connect(self.update_output)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error_occurred.connect(self.handle_error)
        self.processing_thread.start()

    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)

    def update_output(self, text):
        """Update output text"""
        self.output_text.append(text)

    def processing_finished(self, output_path):
        """Handle processing completion"""
        self.start_button.setEnabled(True)
        self.output_text.append(f'\nProcessing completed!\nOutput saved to: {output_path}')
        
        QMessageBox.information(
            self,
            'Processing Complete',
            f'Output saved to:\n{output_path}'
        )

    def handle_error(self, error_message):
        """Handle processing errors"""
        self.start_button.setEnabled(True)
        self.output_text.append(f'\nError: {error_message}')
        
        QMessageBox.critical(
            self,
            'Error',
            f'An error occurred:\n{error_message}'
        )

    def clear_output(self):
        """Clear output text"""
        self.output_text.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())