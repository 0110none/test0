# -*- coding: utf-8 -*-
import csv
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import cv2
import numpy as np
from loguru import logger
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QMessageBox,
    QFileDialog,
)
import yaml
from core.face_detection import FaceDetector
from core.camera_manager import CameraManager
from core.utils import numpy_to_pixmap
from .face_manager import FaceManagerDialog
class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config_path = Path('config/config.yaml')
        self.setWindowTitle(f"{config['app']['name']} v{config['app']['version']}")
        self.setWindowIcon(QIcon(config['app']['logo']))
        self.setGeometry(100, 100, 1200, 800)
        self.face_detector = FaceDetector(config)
        self.camera_manager = CameraManager('config/camera_config.yaml')
        self.media_path: Optional[str] = None
        self.media_type = "camera"
        self.image_frame: Optional[np.ndarray] = None
        self.current_face_count = 0
        self.current_blurred_count = 0
        self.frame_delay_threshold_s = 2.0
        self.frame_loss_threshold = 5
        processing_cfg = self.config.get('processing', {})
        recognition_cfg = self.config.get('recognition', {})
        self.blur_strength_factor = max(0.1, float(processing_cfg.get('blur_strength', 1.0)))
        self.blur_shape = processing_cfg.get('blur_shape', 'rectangle')
        self.blur_target = processing_cfg.get('blur_target', 'unknown')
        self.detection_interval = max(1, int(recognition_cfg.get('detection_interval_frames', 3)))
        self.frame_index = 0
        self.cached_recognized_faces: List[Tuple[Tuple[int, int, int, int], Optional[str], float]] = []
        self.last_tick_ts = time.time()
        self.current_fps = 0.0
        self.metrics = {
            'frames_total': 0,
            'detect_frames': 0,
            'faces_total': 0,
            'blurred_total': 0,
            'events_total': 0,
            'started_at': time.time(),
        }
        self.event_history: List[Dict[str, str]] = []
        self.last_health_status = None
        self.event_log_path = Path(self.config.get('app', {}).get('log_dir', 'logs')) / 'events.log'
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_config_timer = QTimer(self)
        self.save_config_timer.setSingleShot(True)
        self.save_config_timer.timeout.connect(self.persist_runtime_config)
        self.face_detector.load_known_faces(config['app']['known_faces_dir'])
        self.init_ui()
        self.camera_manager.start_camera()
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(30)
        self.record_event('系统启动', f"推理设备: {self.face_detector.actual_device}")
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(18)
        header = QLabel("实时隐私监控")
        header.setObjectName("sectionTitle")
        header.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        main_layout.addWidget(header)
        subtitle = QLabel("单路摄像头 · 已知人脸保持清晰 · 陌生人自动模糊")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        main_layout.addWidget(subtitle)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)
        video_group = QGroupBox("摄像头画面")
        video_layout = QVBoxLayout(video_group)
        video_layout.setSpacing(12)
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(720, 420)
        self.camera_label.setObjectName("cameraFeed")
        video_layout.addWidget(self.camera_label)
        content_layout.addWidget(video_group, 3)
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)
        right_panel.addWidget(self._build_camera_controls())
        right_panel.addWidget(self._build_processing_controls())
        right_panel.addWidget(self._build_status_board())
        content_layout.addLayout(right_panel, 2)
        main_layout.addLayout(content_layout)
        self.status_bar = self.statusBar()
        self.status_label = QLabel("就绪")
        self.status_bar.addPermanentWidget(self.status_label)
        self.setup_menu_bar()
        self.apply_styles()
    def setup_menu_bar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('文件')
        import_media_action = file_menu.addAction('导入图片/视频')
        import_media_action.triggered.connect(self.import_media_file)
        export_media_action = file_menu.addAction('导出处理结果')
        export_media_action.triggered.connect(self.export_processed_media)
        export_action = file_menu.addAction('导出评估报告')
        export_action.triggered.connect(self.export_evaluation_report)
        save_config_action = file_menu.addAction('保存当前参数')
        save_config_action.triggered.connect(self.persist_runtime_config)
        exit_action = file_menu.addAction('退出')
        exit_action.triggered.connect(self.close)
        tools_menu = menubar.addMenu('工具')
        face_manager_action = tools_menu.addAction('人脸管理')
        face_manager_action.triggered.connect(self.open_face_manager)
        view_menu = menubar.addMenu('视图')
        fullscreen_action = view_menu.addAction('切换全屏')
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
    def _build_camera_controls(self) -> QGroupBox:
        camera_group = QGroupBox("摄像头控制")
        camera_group.setObjectName("cameraControls")
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.setSpacing(12)
        camera_name = self.camera_manager.camera.name if self.camera_manager.camera else "未配置摄像头"
        name_label = QLabel(f"当前摄像头：{camera_name}")
        name_label.setObjectName("hintLabel")
        camera_layout.addWidget(name_label)
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("启动")
        self.start_btn.setObjectName("startButton")
        self.start_btn.clicked.connect(self.start_camera_stream)
        btn_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.clicked.connect(self.stop_camera_stream)
        btn_layout.addWidget(self.stop_btn)
        camera_layout.addLayout(btn_layout)
        return camera_group
    def _build_processing_controls(self) -> QGroupBox:
        process_group = QGroupBox("处理参数")
        process_group.setObjectName("processingControls")
        process_layout = QVBoxLayout(process_group)
        process_layout.setSpacing(12)
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("识别阈值："))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 100)
        self.threshold_slider.setValue(int(self.config['recognition']['recognition_threshold'] * 100))
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_value = QLabel(f"{self.threshold_slider.value() / 100:.2f}")
        threshold_layout.addWidget(self.threshold_value)
        process_layout.addLayout(threshold_layout)
        blur_layout = QHBoxLayout()
        blur_layout.addWidget(QLabel("模糊范围："))
        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(50, 200)
        self.blur_slider.setValue(int(self.blur_strength_factor * 100))
        self.blur_slider.valueChanged.connect(self.update_blur_strength)
        blur_layout.addWidget(self.blur_slider)
        self.blur_value = QLabel(f"{self.blur_slider.value() / 100:.2f}x")
        blur_layout.addWidget(self.blur_value)
        process_layout.addLayout(blur_layout)
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("检测间隔帧："))
        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setRange(1, 8)
        self.interval_slider.setValue(self.detection_interval)
        self.interval_slider.valueChanged.connect(self.update_detection_interval)
        interval_layout.addWidget(self.interval_slider)
        self.interval_value = QLabel(f"{self.interval_slider.value()} 帧")
        interval_layout.addWidget(self.interval_value)
        process_layout.addLayout(interval_layout)
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(QLabel("模糊形状："))
        self.blur_shape_group = QButtonGroup(self)
        self.rect_radio = QRadioButton("矩形")
        self.ellipse_radio = QRadioButton("椭圆")
        self.blur_shape_group.addButton(self.rect_radio)
        self.blur_shape_group.addButton(self.ellipse_radio)
        shape_layout.addWidget(self.rect_radio)
        shape_layout.addWidget(self.ellipse_radio)
        shape_layout.addStretch(1)
        self.rect_radio.setChecked(self.blur_shape != 'ellipse')
        self.ellipse_radio.setChecked(self.blur_shape == 'ellipse')
        self.blur_shape_group.buttonClicked.connect(self.update_blur_shape)
        process_layout.addLayout(shape_layout)
        target_layout = QHBoxLayout()
        target_layout.addWidget(QLabel("模糊对象："))
        self.blur_target_group = QButtonGroup(self)
        self.unknown_radio = QRadioButton("仅陌生人")
        self.all_faces_radio = QRadioButton("全部人脸")
        self.blur_target_group.addButton(self.unknown_radio)
        self.blur_target_group.addButton(self.all_faces_radio)
        target_layout.addWidget(self.unknown_radio)
        target_layout.addWidget(self.all_faces_radio)
        target_layout.addStretch(1)
        self.unknown_radio.setChecked(self.blur_target != 'all')
        self.all_faces_radio.setChecked(self.blur_target == 'all')
        self.blur_target_group.buttonClicked.connect(self.update_blur_target)
        process_layout.addLayout(target_layout)
        return process_group
    def _build_status_board(self) -> QGroupBox:
        status_group = QGroupBox("系统状态")
        status_group.setObjectName("systemStatusPanel")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(10)
        self.status_display = QLabel("正在加载状态...")
        self.status_display.setWordWrap(True)
        self.status_display.setObjectName("statusText")
        status_layout.addWidget(self.status_display)
        return status_group
    def open_face_manager(self):
        dialog = FaceManagerDialog(self.face_detector, self.config['app']['known_faces_dir'])
        dialog.exec_()
        self.face_detector.load_known_faces(self.config['app']['known_faces_dir'])
        self.record_event('人脸库', '已刷新已知人脸数据')
    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    def start_camera_stream(self):
        if self.camera_manager.start_camera():
            self.status_label.setText("摄像头已启动")
            self.record_event('摄像头', '手动启动摄像头')
    def stop_camera_stream(self):
        self.camera_manager.stop_camera()
        self.status_label.setText("摄像头已停止")
        self.record_event('摄像头', '手动停止摄像头')
    def update_threshold(self, value):
        threshold = value / 100
        self.face_detector.recognition_threshold = threshold
        self.config.setdefault('recognition', {})['recognition_threshold'] = threshold
        self.threshold_value.setText(f"{threshold:.2f}")
        self.schedule_config_save()
    def update_blur_strength(self, value: int):
        self.blur_strength_factor = max(0.1, value / 100)
        self.config.setdefault('processing', {})['blur_strength'] = self.blur_strength_factor
        self.blur_value.setText(f"{self.blur_strength_factor:.2f}x")
        self.schedule_config_save()
    def update_detection_interval(self, value: int):
        self.detection_interval = max(1, int(value))
        self.config.setdefault('recognition', {})['detection_interval_frames'] = self.detection_interval
        self.interval_value.setText(f"{self.detection_interval} 帧")
        self.schedule_config_save()
    def update_blur_shape(self, _button=None):
        self.blur_shape = 'ellipse' if self.ellipse_radio.isChecked() else 'rectangle'
        self.config.setdefault('processing', {})['blur_shape'] = self.blur_shape
        self.schedule_config_save()
    def update_blur_target(self, _button=None):
        self.blur_target = "all" if self.all_faces_radio.isChecked() else "unknown"
        self.config.setdefault('processing', {})['blur_target'] = self.blur_target
        self.schedule_config_save()
    def schedule_config_save(self):
        self.save_config_timer.start(500)
    def persist_runtime_config(self):
        try:
            with self.config_path.open('w', encoding='utf-8') as f:
                yaml.safe_dump(self.config, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    def update(self):
        try:
            frame = self._get_input_frame()
            total_faces = 0
            total_blurred = 0
            if frame is not None:
                processed_frame, face_count, blurred_count = self.process_frame(frame)
                total_faces += face_count
                total_blurred += blurred_count
                self.display_frame(processed_frame)
                self.metrics['frames_total'] += 1
                self.metrics['faces_total'] += face_count
                self.metrics['blurred_total'] += blurred_count
            self.current_face_count = total_faces
            self.current_blurred_count = total_blurred
            self.current_fps = self._calculate_fps()
            self.status_label.setText(
                f"FPS: {self.current_fps:.1f} | 检测到人脸: {total_faces} | 已模糊: {total_blurred}"
            )
            self.update_status()
        except Exception as e:
            logger.error(f"更新循环错误: {e}")
            self.status_label.setText(f"错误: {str(e)}")
    def _get_input_frame(self) -> Optional[np.ndarray]:
        if self.media_type == "image":
            return None if self.image_frame is None else self.image_frame.copy()
        if self.media_type == "video" and self.media_path:
            if not hasattr(self, "_media_cap") or self._media_cap is None:
                self._media_cap = cv2.VideoCapture(self.media_path)
            ok, frame = self._media_cap.read()
            if not ok:
                self._media_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._media_cap.read()
            return frame if ok else None
        return self.camera_manager.get_frame()
    def import_media_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择图片或视频', '', '媒体文件 (*.png *.jpg *.jpeg *.bmp *.mp4 *.avi *.mov *.mkv)')
        if not file_path:
            return
        ext = Path(file_path).suffix.lower()
        image_exts = {'.png', '.jpg', '.jpeg', '.bmp'}
        if ext in image_exts:
            image = cv2.imread(file_path)
            if image is None:
                QMessageBox.warning(self, '导入失败', '无法读取图片文件')
                return
            self.media_path = file_path
            self.media_type = 'image'
            self.image_frame = image
            if hasattr(self, '_media_cap') and self._media_cap is not None:
                self._media_cap.release()
                self._media_cap = None
            self.status_label.setText('已导入图片')
            self.record_event('媒体', f'导入图片: {Path(file_path).name}')
            return
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            QMessageBox.warning(self, '导入失败', '无法读取视频文件')
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frame_count / max(fps, 1e-6)
        cap.release()
        if duration > 60.0:
            QMessageBox.warning(self, '导入失败', '仅支持导入一分钟以内的视频')
            return
        self.media_path = file_path
        self.media_type = 'video'
        self.image_frame = None
        if hasattr(self, '_media_cap') and self._media_cap is not None:
            self._media_cap.release()
        self._media_cap = cv2.VideoCapture(file_path)
        self.status_label.setText('已导入视频')
        self.record_event('媒体', f'导入视频: {Path(file_path).name} ({duration:.1f}s)')
    def export_processed_media(self):
        if self.media_type == 'camera' or not self.media_path:
            QMessageBox.information(self, '提示', '请先导入图片或视频后再导出')
            return
        if self.media_type == 'image' and self.image_frame is not None:
            out_path, _ = QFileDialog.getSaveFileName(self, '导出图片', 'blurred_output.png', 'PNG 图片 (*.png);;JPG 图片 (*.jpg)')
            if not out_path:
                return
            frame, _, _ = self.process_frame(self.image_frame.copy())
            cv2.imwrite(out_path, frame)
            self.record_event('媒体', f'导出图片: {Path(out_path).name}')
            QMessageBox.information(self, '导出成功', f'处理后图片已导出\n{out_path}')
            return
        out_path, _ = QFileDialog.getSaveFileName(self, '导出视频', 'blurred_output.mp4', 'MP4 视频 (*.mp4)')
        if not out_path:
            return
        cap = cv2.VideoCapture(self.media_path)
        if not cap.isOpened():
            QMessageBox.warning(self, '导出失败', '无法读取原视频文件')
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        old_index = self.frame_index
        old_cache = list(self.cached_recognized_faces)
        self.frame_index = 0
        self.cached_recognized_faces = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed, _, _ = self.process_frame(frame)
            writer.write(processed)
        cap.release()
        writer.release()
        self.frame_index = old_index
        self.cached_recognized_faces = old_cache
        self.record_event('媒体', f'导出视频: {Path(out_path).name}')
        QMessageBox.information(self, '导出成功', f'处理后视频已导出\n{out_path}')
    def _calculate_fps(self) -> float:
        now = time.time()
        gap = max(1e-6, now - self.last_tick_ts)
        self.last_tick_ts = now
        return 1.0 / gap
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        processed_frame = frame.copy()
        self.frame_index += 1
        should_detect = (
            self.frame_index % self.detection_interval == 0
            or not self.cached_recognized_faces
        )
        if should_detect:
            try:
                faces = self.face_detector.detect_faces(frame)
                recognized_faces = self.face_detector.recognize_faces(faces)
                self.cached_recognized_faces = []
                for face, known_face, confidence in recognized_faces:
                    clipped_bbox = self._clip_bbox(face.bbox, processed_frame.shape)
                    if clipped_bbox is None:
                        continue
                    self.cached_recognized_faces.append((clipped_bbox, known_face.name if known_face else None, confidence))
                self.metrics['detect_frames'] += 1
            except Exception as e:
                logger.error(f"检测人脸失败: {e}")
                return processed_frame, 0, 0
        blurred_count = 0
        for bbox, known_name, confidence in self.cached_recognized_faces:
            if known_name and self.blur_target != "all":
                self._draw_known_face(processed_frame, bbox, known_name, confidence)
            else:
                if self._blur_face_region(processed_frame, bbox):
                    blurred_count += 1
        return processed_frame, len(self.cached_recognized_faces), blurred_count
    def _clip_bbox(self, bbox: np.ndarray, frame_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        try:
            h, w = frame_shape[:2]
            x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                return None
            return x1, y1, x2, y2
        except Exception as e:
            logger.error(f"裁剪人脸框失败: {e}")
            return None
    def _draw_known_face(self, image: np.ndarray, bbox: Tuple[int, int, int, int], name: str, confidence: float) -> None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        try:
            name.encode('ascii')
            label_text = f"{name} {confidence:.2f}"
        except UnicodeEncodeError:
            label_text = f"Known {confidence:.2f}"
        text_org = (x1, y1 - 10 if y1 - 10 > 10 else y2 + 20)
        cv2.putText(image, label_text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    def _blur_face_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = bbox
        face_region = image[y1:y2, x1:x2]
        if face_region.size == 0:
            return False
        kernel = self._calculate_blur_kernel(x2 - x1, y2 - y1)
        try:
            blurred = cv2.GaussianBlur(face_region, (kernel, kernel), 0)
            if self.blur_shape == 'ellipse':
                mask = np.zeros_like(face_region)
                center = (face_region.shape[1] // 2, face_region.shape[0] // 2)
                axes = (face_region.shape[1] // 2, face_region.shape[0] // 2)
                cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
                image[y1:y2, x1:x2] = np.where(mask > 0, blurred, face_region)
            else:
                image[y1:y2, x1:x2] = blurred
            return True
        except Exception as e:
            logger.error(f"模糊陌生人脸失败: {e}")
            return False
    def _calculate_blur_kernel(self, width: int, height: int) -> int:
        base = max(width, height) // 6
        kernel = max(15, base * 2 + 1)
        scaled_kernel = int(round(kernel * self.blur_strength_factor))
        if scaled_kernel % 2 == 0:
            scaled_kernel += 1
        return max(3, scaled_kernel)
    def display_frame(self, frame: np.ndarray):
        try:
            if frame is None:
                return
            pixmap = numpy_to_pixmap(frame)
            scaled = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled)
        except Exception as e:
            logger.error(f"显示帧错误: {e}")
    def update_status(self):
        try:
            status_text = []
            status_text.append("=== 摄像头状态 ===")
            camera_status = self.camera_manager.get_camera_status()
            if camera_status:
                running = '运行中' if camera_status['running'] else '已停止'
                status_text.append(f"{camera_status['name']}：{running}")
            else:
                status_text.append("未加载摄像头配置")
            health = self.camera_manager.get_camera_health(
                delay_threshold=self.frame_delay_threshold_s,
                loss_threshold=self.frame_loss_threshold,
            )
            if health.get('last_frame_gap_s') is not None:
                status_text.append(f"最近帧间隔：{health['last_frame_gap_s']:.2f}s")
            if health.get('delay_warning'):
                status_text.append("⚠️ 摄像头画面存在延迟")
            if health.get('loss_warning'):
                status_text.append("❌ 摄像头信号可能丢失，请检查连接")
            status_text.append(f"当前状态：{health.get('status', '未知')}")
            current_health_status = health.get('status', '未知')
            if self.last_health_status != current_health_status:
                self.record_event('摄像头状态', f'状态变更为: {current_health_status}')
                self.last_health_status = current_health_status
            status_text.append("\n=== 推理状态 ===")
            status_text.append(f"推理设备：{self.face_detector.actual_device}")
            status_text.append(f"实时 FPS：{self.current_fps:.1f}")
            status_text.append(f"检测抽样：每 {self.detection_interval} 帧检测一次")
            status_text.append("\n=== 人脸库 ===")
            status_text.append(f"已知人脸数量：{len(self.face_detector.known_faces)}")
            status_text.append("\n=== 实时统计 ===")
            status_text.append(f"当前检测到的人脸数量：{self.current_face_count}")
            status_text.append(f"当前被模糊的人脸数量：{self.current_blurred_count}")
            status_text.append(f"当前模糊范围倍率：{self.blur_strength_factor:.2f}x")
            status_text.append(f"模糊形状：{'椭圆' if self.blur_shape == 'ellipse' else '矩形'}")
            status_text.append("\n=== 最近事件 ===")
            if self.event_history:
                for event in self.event_history[-4:]:
                    status_text.append(f"[{event['time']}] {event['type']} - {event['message']}")
            else:
                status_text.append("暂无事件")
            self.status_display.setText("\n".join(status_text))
        except Exception as e:
            logger.error(f"更新状态失败: {e}")
    def record_event(self, event_type: str, message: str):
        ts = time.strftime('%H:%M:%S')
        item = {'time': ts, 'type': event_type, 'message': message}
        self.event_history.append(item)
        self.metrics['events_total'] += 1
        if len(self.event_history) > 80:
            self.event_history = self.event_history[-80:]
        with self.event_log_path.open('a', encoding='utf-8') as f:
            f.write(f"[{ts}] {event_type}: {message}\n")
    def export_evaluation_report(self):
        try:
            report_dir = Path(self.config.get('app', {}).get('log_dir', 'logs'))
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"evaluation_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            frames = max(1, self.metrics['frames_total'])
            uptime = max(1e-6, time.time() - self.metrics['started_at'])
            avg_fps = self.metrics['frames_total'] / uptime
            avg_faces = self.metrics['faces_total'] / frames
            blur_ratio = self.metrics['blurred_total'] / max(1, self.metrics['faces_total'])
            with report_path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['metric', 'value'])
                writer.writerow(['frames_total', self.metrics['frames_total']])
                writer.writerow(['detect_frames', self.metrics['detect_frames']])
                writer.writerow(['avg_fps', f"{avg_fps:.2f}"])
                writer.writerow(['avg_faces_per_frame', f"{avg_faces:.3f}"])
                writer.writerow(['blur_ratio', f"{blur_ratio:.3f}"])
                writer.writerow(['recognition_threshold', self.face_detector.recognition_threshold])
                writer.writerow(['blur_strength', self.blur_strength_factor])
                writer.writerow(['detection_interval_frames', self.detection_interval])
                writer.writerow(['device', self.face_detector.actual_device])
                writer.writerow(['events_total', self.metrics['events_total']])
            self.record_event('评估', f"导出评估报告: {report_path.name}")
            QMessageBox.information(self, '导出成功', f'评估报告已导出:\n{report_path}')
        except Exception as e:
            logger.error(f"导出评估报告失败: {e}")
            QMessageBox.warning(self, '导出失败', str(e))
    def closeEvent(self, event):
        try:
            self.persist_runtime_config()
            if hasattr(self, "_media_cap") and self._media_cap is not None:
                self._media_cap.release()
            self.camera_manager.stop_camera()
            self.update_timer.stop()
            event.accept()
        except Exception as e:
            logger.error(f"关闭程序时出错: {e}")
            event.accept()
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #060a14;
                color: #e2e8f0;
            }
            QLabel {
                color: #dbe7ff;
            }
            QLabel#sectionTitle {
                font-size: 24px;
                font-weight: 700;
                padding: 6px 0 2px 0;
                color: #7dd3fc;
                letter-spacing: 1px;
            }
            QLabel#subtitle {
                color: #a5b4fc;
                padding-bottom: 12px;
            }
            QLabel#hintLabel {
                color: #c7d2fe;
            }
            QLabel#statusText {
                color: #e2e8f0;
                line-height: 1.5em;
                background: rgba(15, 23, 42, 0.4);
                border: 1px solid rgba(99, 102, 241, 0.22);
                border-radius: 10px;
                padding: 10px;
            }
            QLabel#cameraFeed {
                background-color: rgba(10, 18, 35, 0.92);
                border-radius: 16px;
                border: 2px solid rgba(34, 211, 238, 0.35);
            }
            QGroupBox {
                border: 1px solid rgba(99, 102, 241, 0.45);
                border-radius: 14px;
                margin-top: 10px;
                padding: 16px;
                font-weight: 700;
                background-color: rgba(19, 26, 46, 0.86);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                color: #bfdbfe;
            }
            QGroupBox#cameraControls {
                background-color: rgba(17, 43, 68, 0.72);
                border: 1px solid rgba(56, 189, 248, 0.5);
            }
            QGroupBox#processingControls {
                background-color: rgba(39, 32, 72, 0.78);
                border: 1px solid rgba(129, 140, 248, 0.58);
            }
            QGroupBox#systemStatusPanel {
                background-color: rgba(20, 39, 56, 0.82);
                border: 1px solid rgba(45, 212, 191, 0.52);
            }
            QPushButton {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0284c7, stop:1 #4f46e5
                );
                border: 1px solid rgba(165, 180, 252, 0.8);
                border-radius: 12px;
                padding: 10px 16px;
                color: #f8fafc;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0ea5e9, stop:1 #6366f1
                );
                border: 1px solid #c4b5fd;
            }
            QPushButton:pressed {
                background-color: #312e81;
            }
            QPushButton#startButton {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #059669, stop:1 #0ea5a4);
                border: 1px solid rgba(110, 231, 183, 0.85);
            }
            QPushButton#startButton:hover {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #10b981, stop:1 #14b8a6);
            }
            QPushButton#stopButton {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #be123c, stop:1 #7c3aed);
                border: 1px solid rgba(251, 113, 133, 0.88);
            }
            QPushButton#stopButton:hover {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #e11d48, stop:1 #8b5cf6);
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: rgba(30, 41, 59, 0.88);
                border: 1px solid rgba(129, 140, 248, 0.35);
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #22d3ee, stop:1 #818cf8);
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #f8fafc;
                border: 2px solid #22d3ee;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QRadioButton {
                spacing: 8px;
                color: #dbeafe;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
            QRadioButton::indicator:unchecked {
                border: 1px solid rgba(148, 163, 184, 0.8);
                background: rgba(15, 23, 42, 0.9);
                border-radius: 7px;
            }
            QRadioButton::indicator:checked {
                border: 1px solid #38bdf8;
                background: #22d3ee;
                border-radius: 7px;
            }
        """)
