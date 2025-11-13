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
    QTabWidget,
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

        # 模糊参数（范围与强度）
        processing_cfg = self.config.get('processing', {})
        self.blur_range_factor = float(processing_cfg.get('blur_range', 1.0))
        if self.blur_range_factor <= 0:
            self.blur_range_factor = 1.0

        self.blur_strength_factor = float(processing_cfg.get('blur_strength', 1.0))
        if self.blur_strength_factor < 0:
            self.blur_strength_factor = 0.0

        # 加载已知人脸库
        self.face_detector.load_known_faces(config['app']['known_faces_dir'])

        # --- 初始化 UI ---
        self.init_ui()

        # 启动摄像头线程
        self.camera_manager.start_all_cameras()

        # 启动定时更新器（刷新画面）
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(30)  # 约 30 FPS

    # ------------------------------
    # 初始化与 UI 构建部分
    # ------------------------------
    def init_ui(self):
        """设置主界面布局：Tab页 + 状态栏 + 菜单栏"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 标签页（Tab）
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 添加功能页
        self.setup_monitor_tab()   # 摄像头监控界面
        self.setup_controls_tab()  # 控制参数界面

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

    def setup_monitor_tab(self):
        """监控界面：显示摄像头实时画面"""
        monitor_tab = QWidget()
        self.tab_widget.addTab(monitor_tab, "监控")

        layout = QVBoxLayout(monitor_tab)
        title = QLabel("实时监控")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(title)

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 360)
        self.camera_label.setObjectName("cameraFeed")
        layout.addWidget(self.camera_label, 1)

    def setup_controls_tab(self):
        """控制界面：用于调节识别阈值、处理间隔、摄像头启停等"""
        controls_tab = QWidget()
        self.tab_widget.addTab(controls_tab, "控制")
        layout = QVBoxLayout(controls_tab)

        camera_group = QGroupBox("摄像头控制")
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.setSpacing(12)

        camera_name = self.camera_manager.camera.name if self.camera_manager.camera else "未配置摄像头"
        name_label = QLabel(f"当前摄像头：{camera_name}")
        camera_layout.addWidget(name_label)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("启动摄像头")
        self.start_btn.clicked.connect(self.start_camera_stream)
        btn_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止摄像头")
        self.stop_btn.clicked.connect(self.stop_camera_stream)
        btn_layout.addWidget(self.stop_btn)
        camera_layout.addLayout(btn_layout)

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
        camera_layout.addLayout(threshold_layout)

        blur_range_layout = QHBoxLayout()
        blur_range_label = QLabel("模糊范围：")
        blur_range_layout.addWidget(blur_range_label)

        self.blur_range_slider = QSlider(Qt.Horizontal)
        self.blur_range_slider.setRange(50, 250)
        initial_range_value = int(self.blur_range_factor * 100)
        initial_range_value = max(self.blur_range_slider.minimum(), min(self.blur_range_slider.maximum(), initial_range_value))
        self.blur_range_slider.setValue(initial_range_value)
        self.blur_range_slider.valueChanged.connect(self.update_blur_range)
        blur_range_layout.addWidget(self.blur_range_slider)

        self.blur_range_value = QLabel(f"{self.blur_range_slider.value() / 100:.2f}x")
        blur_range_layout.addWidget(self.blur_range_value)
        self.update_blur_range(self.blur_range_slider.value())
        camera_layout.addLayout(blur_range_layout)

        blur_strength_layout = QHBoxLayout()
        blur_strength_label = QLabel("模糊强度：")
        blur_strength_layout.addWidget(blur_strength_label)

        self.blur_strength_slider = QSlider(Qt.Horizontal)
        self.blur_strength_slider.setRange(0, 300)
        initial_strength_value = int(self.blur_strength_factor * 100)
        initial_strength_value = max(self.blur_strength_slider.minimum(), min(self.blur_strength_slider.maximum(), initial_strength_value))
        self.blur_strength_slider.setValue(initial_strength_value)
        self.blur_strength_slider.valueChanged.connect(self.update_blur_strength)
        blur_strength_layout.addWidget(self.blur_strength_slider)

        self.blur_strength_value = QLabel(f"{self.blur_strength_slider.value() / 100:.2f}x")
        blur_strength_layout.addWidget(self.blur_strength_value)
        self.update_blur_strength(self.blur_strength_slider.value())
        camera_layout.addLayout(blur_strength_layout)

        layout.addWidget(camera_group)

        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout(status_group)
        status_layout.setSpacing(8)
        self.status_display = QLabel("正在加载状态...")
        self.status_display.setWordWrap(True)
        status_layout.addWidget(self.status_display)
        layout.addWidget(status_group)

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

    def update_blur_range(self, value: int):
        """调整模糊范围（扩大待模糊区域）"""
        self.blur_range_factor = max(0.1, value / 100)
        if 'processing' not in self.config:
            self.config['processing'] = {}
        self.config['processing']['blur_range'] = self.blur_range_factor
        if hasattr(self, 'blur_range_value'):
            self.blur_range_value.setText(f"{self.blur_range_factor:.2f}x")

    def update_blur_strength(self, value: int):
        """调整模糊强度（缩放高斯模糊核大小）"""
        self.blur_strength_factor = max(0.0, value / 100)
        if 'processing' not in self.config:
            self.config['processing'] = {}
        self.config['processing']['blur_strength'] = self.blur_strength_factor
        if hasattr(self, 'blur_strength_value'):
            self.blur_strength_value.setText(f"{self.blur_strength_factor:.2f}x")

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
                expanded_bbox = self._expand_bbox(clipped_bbox, processed_frame.shape, self.blur_range_factor)
                if expanded_bbox is None:
                    expanded_bbox = clipped_bbox
                if self._blur_face_region(processed_frame, expanded_bbox):
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

        if self.blur_strength_factor <= 0:
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

    def _expand_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, int],
        factor: float,
    ) -> Optional[Tuple[int, int, int, int]]:
        """按指定倍率扩展人脸框并限制在图像范围内"""
        if factor <= 1.0:
            return bbox

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None

        cx = x1 + width / 2
        cy = y1 + height / 2
        new_width = width * factor
        new_height = height * factor

        new_x1 = int(round(cx - new_width / 2))
        new_y1 = int(round(cy - new_height / 2))
        new_x2 = int(round(cx + new_width / 2))
        new_y2 = int(round(cy + new_height / 2))

        return self._clip_bbox((new_x1, new_y1, new_x2, new_y2), frame_shape)

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
            status_text.append(f"当前模糊范围倍率：{self.blur_range_factor:.2f}x")
            status_text.append(f"当前模糊强度倍率：{self.blur_strength_factor:.2f}x")

            self.status_display.setText("\n".join(status_text))

        except Exception as e:
            logger.error(f"更新状态失败: {e}")

    def closeEvent(self, event):
        """程序退出时释放资源：停止摄像头、定时器"""
        try:
            self.camera_manager.stop_all_cameras()
            self.update_timer.stop()
            event.accept()
        except Exception as e:
            logger.error(f"关闭程序时出错: {e}")
            event.accept()

    def apply_styles(self):
        """统一设置应用的样式和色彩风格"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a;
                color: #e2e8f0;
            }
            QLabel {
                color: #e2e8f0;
            }
            QLabel#sectionTitle {
                font-size: 20px;
                font-weight: 600;
                padding: 8px 0;
                color: #93c5fd;
            }
            QLabel#cameraFeed {
                background-color: #1e293b;
                border-radius: 12px;
                border: 2px solid #1d4ed8;
            }
            QTabWidget::pane {
                border: 1px solid #1d4ed8;
                border-radius: 10px;
                background-color: #111c34;
            }
            QTabBar::tab {
                padding: 10px 24px;
                background-color: transparent;
                color: #cbd5f5;
                border-bottom: 2px solid transparent;
            }
            QTabBar::tab:selected {
                background-color: #1e3a8a;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                color: #ffffff;
                border-bottom: 2px solid #60a5fa;
            }
            QPushButton {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #2563eb, stop:1 #1e40af);
                border: 2px solid #60a5fa;
                border-radius: 10px;
                padding: 10px 20px;
                color: #ffffff;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #3b82f6, stop:1 #1d4ed8);
                border: 2px solid #93c5fd;
            }
            QPushButton:pressed {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #1d4ed8, stop:1 #1e3a8a);
                border: 2px solid #3b82f6;
                padding-top: 12px;
                padding-bottom: 8px;
            }
            QGroupBox {
                border: 1px solid #1d4ed8;
                border-radius: 12px;
                margin-top: 12px;
                padding: 16px;
                font-weight: 600;
                background-color: rgba(30, 64, 175, 0.35);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                color: #bfdbfe;
            }
        """)
