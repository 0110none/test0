# -*- coding: utf-8 -*-

import sys
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QGroupBox,
)

# 导入核心模块
from core.face_detection import FaceDetector
from core.camera_manager import CameraManager
from core.utils import numpy_to_pixmap

# 导入子界面模块
from .face_manager import FaceManagerDialog


class MainWindow(QMainWindow):
    """
    系统主窗口（MainWindow）
    ------------------------
    负责整合摄像头流、人脸检测与识别模块，并在界面上呈现隐私保护后的画面。
    已注册人脸将保持清晰，陌生人脸会被自动模糊处理。
    """

    def __init__(self, config):
        """初始化主界面及所有核心组件"""
        super().__init__()
        self.config = config
        self.setWindowTitle(f"{config['app']['name']} v{config['app']['version']}")
        self.setWindowIcon(QIcon(config['app']['logo']))
        self.setGeometry(100, 100, 1200, 800)

        # --- 初始化核心组件 ---
        self.face_detector = FaceDetector(config)                         # 人脸检测与识别模块
        self.camera_manager = CameraManager('config/camera_config.yaml')  # 摄像头管理模块

        # 实时统计信息
        self.current_face_count = 0
        self.current_blurred_count = 0

        # 模糊强度（用于控制高斯模糊范围）
        processing_cfg = self.config.get('processing', {})
        self.blur_strength_factor = float(processing_cfg.get('blur_strength', 1.0))
        if self.blur_strength_factor <= 0:
            self.blur_strength_factor = 1.0

        # 加载已知人脸库
        self.face_detector.load_known_faces(config['app']['known_faces_dir'])

        # --- 初始化 UI ---
        self.init_ui()

        # 启动摄像头线程
        self.camera_manager.start_camera()

        # 启动定时更新器（刷新画面）
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(30)  # 约 30 FPS

    # ------------------------------
    # 初始化与 UI 构建部分
    # ------------------------------
    def init_ui(self):
        """设置主界面布局：单摄像头卡片式展示 + 控制面板"""
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

        # 左侧：视频窗口
        video_group = QGroupBox("摄像头画面")
        video_layout = QVBoxLayout(video_group)
        video_layout.setSpacing(12)
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(720, 420)
        self.camera_label.setObjectName("cameraFeed")
        video_layout.addWidget(self.camera_label)
        content_layout.addWidget(video_group, 3)

        # 右侧：控制面板
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)
        right_panel.addWidget(self._build_camera_controls())
        right_panel.addWidget(self._build_processing_controls())
        right_panel.addWidget(self._build_status_board())
        content_layout.addLayout(right_panel, 2)

        main_layout.addLayout(content_layout)

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_label = QLabel("就绪")
        self.status_bar.addPermanentWidget(self.status_label)

        # 菜单栏
        self.setup_menu_bar()

        # 统一应用样式
        self.apply_styles()

    def setup_menu_bar(self):
        """创建菜单栏（文件、工具、视图）"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')
        exit_action = file_menu.addAction('退出')
        exit_action.triggered.connect(self.close)

        # 工具菜单
        tools_menu = menubar.addMenu('工具')
        face_manager_action = tools_menu.addAction('人脸管理')
        face_manager_action.triggered.connect(self.open_face_manager)

        # 视图菜单
        view_menu = menubar.addMenu('视图')
        fullscreen_action = view_menu.addAction('切换全屏')
        fullscreen_action.triggered.connect(self.toggle_fullscreen)

    def _build_camera_controls(self) -> QGroupBox:
        """摄像头启停控制卡片"""
        camera_group = QGroupBox("摄像头控制")
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.setSpacing(12)

        camera_name = self.camera_manager.camera.name if self.camera_manager.camera else "未配置摄像头"
        name_label = QLabel(f"当前摄像头：{camera_name}")
        name_label.setObjectName("hintLabel")
        camera_layout.addWidget(name_label)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("启动")
        self.start_btn.clicked.connect(self.start_camera_stream)
        btn_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_camera_stream)
        btn_layout.addWidget(self.stop_btn)
        camera_layout.addLayout(btn_layout)

        return camera_group

    def _build_processing_controls(self) -> QGroupBox:
        """阈值与模糊参数调节卡片"""
        process_group = QGroupBox("处理参数")
        process_layout = QVBoxLayout(process_group)
        process_layout.setSpacing(12)

        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("识别阈值：")
        threshold_layout.addWidget(threshold_label)

        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(50, 100)
        self.threshold_slider.setValue(int(self.config['recognition']['recognition_threshold'] * 100))
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_slider)

        self.threshold_value = QLabel(f"{self.threshold_slider.value() / 100:.2f}")
        threshold_layout.addWidget(self.threshold_value)
        process_layout.addLayout(threshold_layout)

        blur_layout = QHBoxLayout()
        blur_label = QLabel("模糊范围：")
        blur_layout.addWidget(blur_label)

        self.blur_slider = QSlider(Qt.Horizontal)
        self.blur_slider.setRange(50, 200)
        initial_blur_value = int(self.blur_strength_factor * 100)
        initial_blur_value = max(self.blur_slider.minimum(), min(self.blur_slider.maximum(), initial_blur_value))
        self.blur_slider.setValue(initial_blur_value)
        self.blur_slider.valueChanged.connect(self.update_blur_strength)
        blur_layout.addWidget(self.blur_slider)

        self.blur_value = QLabel(f"{self.blur_slider.value() / 100:.2f}x")
        blur_layout.addWidget(self.blur_value)
        self.update_blur_strength(self.blur_slider.value())
        process_layout.addLayout(blur_layout)

        return process_group

    def _build_status_board(self) -> QGroupBox:
        """显示运行状态的卡片"""
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(10)
        self.status_display = QLabel("正在加载状态...")
        self.status_display.setWordWrap(True)
        self.status_display.setObjectName("statusText")
        status_layout.addWidget(self.status_display)
        return status_group

    # ------------------------------
    # 菜单动作
    # ------------------------------
    def open_face_manager(self):
        """打开人脸管理窗口"""
        dialog = FaceManagerDialog(self.face_detector, self.config['app']['known_faces_dir'])
        dialog.exec_()
        self.face_detector.load_known_faces(self.config['app']['known_faces_dir'])

    def toggle_fullscreen(self):
        """切换全屏模式"""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # ------------------------------
    # 摄像头控制与参数调节
    # ------------------------------
    def start_camera_stream(self):
        """启动摄像头"""
        if self.camera_manager.start_camera():
            self.status_label.setText("摄像头已启动")

    def stop_camera_stream(self):
        """停止摄像头"""
        self.camera_manager.stop_camera()
        self.status_label.setText("摄像头已停止")

    def update_threshold(self, value):
        """调整识别置信度阈值"""
        threshold = value / 100
        self.face_detector.recognition_threshold = threshold
        self.threshold_value.setText(f"{threshold:.2f}")

    def update_blur_strength(self, value: int):
        """调整模糊强度（缩放高斯模糊核大小）"""
        self.blur_strength_factor = max(0.1, value / 100)
        if 'processing' not in self.config:
            self.config['processing'] = {}
        self.config['processing']['blur_strength'] = self.blur_strength_factor
        self.blur_value.setText(f"{self.blur_strength_factor:.2f}x")

    # ------------------------------
    # 主循环与图像处理
    # ------------------------------
    def update(self):
        """主循环（每30ms执行一次）：获取帧→识别→模糊→显示→更新状态"""
        try:
            frame = self.camera_manager.get_frame()
            total_faces = 0
            total_blurred = 0

            if frame is not None:
                processed_frame, face_count, blurred_count = self.process_frame(frame)
                total_faces += face_count
                total_blurred += blurred_count
                self.display_frame(processed_frame)

            self.current_face_count = total_faces
            self.current_blurred_count = total_blurred
            self.status_label.setText(
                f"检测到人脸: {total_faces} | 已模糊: {total_blurred}"
            )
            self.update_status()

        except Exception as e:
            logger.error(f"更新循环错误: {e}")
            self.status_label.setText(f"错误: {str(e)}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """检测并处理一帧画面，返回处理结果及统计信息"""
        processed_frame = frame.copy()

        try:
            faces = self.face_detector.detect_faces(frame)
        except Exception as e:
            logger.error(f"检测人脸失败: {e}")
            return processed_frame, 0, 0

        if not faces:
            return processed_frame, 0, 0

        recognized_faces = self.face_detector.recognize_faces(faces)
        blurred_count = 0

        for face, known_face, confidence in recognized_faces:
            clipped_bbox = self._clip_bbox(face.bbox, processed_frame.shape)
            if clipped_bbox is None:
                continue

            if known_face:
                self._draw_known_face(processed_frame, clipped_bbox, known_face.name, confidence)
            else:
                if self._blur_face_region(processed_frame, clipped_bbox):
                    blurred_count += 1

        return processed_frame, len(faces), blurred_count

    def _clip_bbox(self, bbox: np.ndarray, frame_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """将人脸框裁剪到图像范围内"""
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
        """在图像上标注已注册人脸"""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        try:
            name.encode('ascii')
            label_text = f"{name} {confidence:.2f}"
        except UnicodeEncodeError:
            label_text = f"Known {confidence:.2f}"

        text_org = (x1, y1 - 10 if y1 - 10 > 10 else y2 + 20)
        cv2.putText(
            image,
            label_text,
            text_org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def _blur_face_region(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """对指定区域进行模糊处理"""
        x1, y1, x2, y2 = bbox
        face_region = image[y1:y2, x1:x2]
        if face_region.size == 0:
            return False

        kernel = self._calculate_blur_kernel(x2 - x1, y2 - y1)
        try:
            blurred = cv2.GaussianBlur(face_region, (kernel, kernel), 0)
            image[y1:y2, x1:x2] = blurred
            return True
        except Exception as e:
            logger.error(f"模糊陌生人脸失败: {e}")
            return False

    def _calculate_blur_kernel(self, width: int, height: int) -> int:
        """根据人脸尺寸自适应计算高斯模糊核大小（保持为奇数）"""
        base = max(width, height) // 6
        kernel = max(15, base * 2 + 1)
        scaled_kernel = int(round(kernel * self.blur_strength_factor))
        if scaled_kernel % 2 == 0:
            scaled_kernel += 1
        return max(3, scaled_kernel)

    def display_frame(self, frame: np.ndarray):
        """将处理后的画面显示到摄像头窗口"""
        try:
            if frame is None:
                return
            pixmap = numpy_to_pixmap(frame)
            if pixmap is None:
                return
            scaled = pixmap.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.camera_label.setPixmap(scaled)
        except Exception as e:
            logger.error(f"显示帧错误: {e}")

    def update_status(self):
        """更新系统状态信息：摄像头、人脸库、模糊统计"""
        try:
            status_text = []

            status_text.append("=== 摄像头状态 ===")
            camera_status = self.camera_manager.get_camera_status()
            if camera_status:
                running = '运行中' if camera_status['running'] else '已停止'
                status_text.append(f"{camera_status['name']}：{running}")
            else:
                status_text.append("未加载摄像头配置")

            status_text.append("\n=== 人脸库 ===")
            status_text.append(f"已知人脸数量：{len(self.face_detector.known_faces)}")

            status_text.append("\n=== 实时统计 ===")
            status_text.append(f"当前检测到的人脸数量：{self.current_face_count}")
            status_text.append(f"当前被模糊的人脸数量：{self.current_blurred_count}")
            status_text.append(f"当前模糊范围倍率：{self.blur_strength_factor:.2f}x")

            self.status_display.setText("\n".join(status_text))

        except Exception as e:
            logger.error(f"更新状态失败: {e}")

    def closeEvent(self, event):
        """程序退出时释放资源：停止摄像头、定时器"""
        try:
            self.camera_manager.stop_camera()
            self.update_timer.stop()
            event.accept()
        except Exception as e:
            logger.error(f"关闭程序时出错: {e}")
            event.accept()

    def apply_styles(self):
        """统一设置应用的样式和色彩风格"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #070b16;
                color: #e2e8f0;
            }
            QLabel {
                color: #e2e8f0;
            }
            QLabel#sectionTitle {
                font-size: 22px;
                font-weight: 700;
                padding: 6px 0 2px 0;
                color: #67e8f9;
                letter-spacing: 1px;
            }
            QLabel#subtitle {
                color: #a5b4fc;
                padding-bottom: 12px;
            }
            QLabel#hintLabel {
                color: #cbd5f5;
            }
            QLabel#statusText {
                color: #e5e7eb;
                line-height: 1.5em;
            }
            QLabel#cameraFeed {
                background-color: rgba(15,23,42,0.75);
                border-radius: 16px;
                border: 2px solid rgba(14,165,233,0.4);
                box-shadow: 0 0 18px rgba(6,182,212,0.35);
            }
            QGroupBox {
                border: 1px solid rgba(99,102,241,0.6);
                border-radius: 14px;
                margin-top: 8px;
                padding: 16px;
                font-weight: 600;
                background-color: rgba(24, 30, 54, 0.85);
                box-shadow: inset 0 0 18px rgba(79,70,229,0.35);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                color: #93c5fd;
                background-color: transparent;
            }
            QPushButton {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #0ea5e9, stop:1 #6366f1);
                border: 1px solid #67e8f9;
                border-radius: 12px;
                padding: 12px 18px;
                color: #ffffff;
                font-weight: 700;
                letter-spacing: 0.5px;
                box-shadow: 0 0 12px rgba(103,102,241,0.35);
            }
            QPushButton:hover {
                border: 1px solid #a5b4fc;
                box-shadow: 0 0 14px rgba(103,102,241,0.55);
            }
            QPushButton:pressed {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #312e81, stop:1 #0b7285);
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: rgba(99,102,241,0.35);
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #67e8f9;
                border: 1px solid #0ea5e9;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
                box-shadow: 0 0 10px rgba(14,165,233,0.65);
            }
        """)
