# -*- coding: utf-8 -*-

import sys
from typing import Dict, Optional, Tuple

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
    QScrollArea,
    QGridLayout,
    QMessageBox,
    QFileDialog,
    QComboBox,
    QSlider,
    QFrame,
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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        # 摄像头画面容器
        self.camera_container = QWidget()
        self.camera_grid = QGridLayout(self.camera_container)
        self.camera_grid.setSpacing(10)
        scroll.setWidget(self.camera_container)

        layout = QVBoxLayout(monitor_tab)
        title = QLabel("实时监控")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(title)
        layout.addWidget(scroll)

        # 为每个摄像头添加显示区域
        self.camera_labels: Dict[int, QLabel] = {}
        self.camera_cards = []
        for cam_id in sorted(self.camera_manager.cameras.keys()):
            self.add_camera_display(cam_id)

    def add_camera_display(self, cam_id: int):
        """为指定摄像头创建显示卡片"""
        if cam_id in self.camera_labels:
            return

        cam_config = self.camera_manager.cameras.get(cam_id)
        if cam_config is None:
            return

        card = QFrame()
        card.setObjectName("cameraCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(8)

        header = QLabel(cam_config.name)
        header.setAlignment(Qt.AlignCenter)
        header.setObjectName("cameraTitle")
        card_layout.addWidget(header)

        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(360, 220)
        label.setObjectName("cameraFeed")
        card_layout.addWidget(label)

        self.camera_labels[cam_id] = label
        self.camera_cards.append(card)

        index = len(self.camera_cards) - 1
        row = index // 2
        col = index % 2
        self.camera_grid.addWidget(card, row, col)

    def refresh_camera_combo(self):
        """刷新摄像头下拉框内容"""
        if not hasattr(self, 'camera_combo'):
            return

        current_id = self.camera_combo.currentData() if self.camera_combo.count() else None
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()

        for cam_id in sorted(self.camera_manager.cameras.keys()):
            cam_config = self.camera_manager.cameras[cam_id]
            self.camera_combo.addItem(f"摄像头 {cam_id}: {cam_config.name}", cam_id)

        self.camera_combo.blockSignals(False)

        if current_id is not None:
            index = self.camera_combo.findData(current_id)
            if index != -1:
                self.camera_combo.setCurrentIndex(index)

        if self.camera_combo.count() > 0 and self.camera_combo.currentIndex() == -1:
            self.camera_combo.setCurrentIndex(0)

    def add_video_source(self):
        """选择并添加本地视频文件作为新的监控源"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv *.wmv);;所有文件 (*)",
        )

        if not file_path:
            return

        new_id = self.camera_manager.add_video_source(file_path)
        if new_id is None:
            QMessageBox.warning(self, "错误", "无法加载所选视频文件")
            return

        self.add_camera_display(new_id)
        self.refresh_camera_combo()
        self.status_label.setText(f"已添加视频源: {file_path}")

    def setup_controls_tab(self):
        """控制界面：用于调节识别阈值、处理间隔、摄像头启停等"""
        controls_tab = QWidget()
        self.tab_widget.addTab(controls_tab, "控制")
        layout = QVBoxLayout(controls_tab)

        # 摄像头控制区
        camera_group = QGroupBox("摄像头控制")
        camera_layout = QVBoxLayout(camera_group)
        camera_layout.setSpacing(12)

        self.camera_combo = QComboBox()
        camera_layout.addWidget(self.camera_combo)
        self.refresh_camera_combo()

        # 启动/停止按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("启动摄像头")
        self.start_btn.clicked.connect(self.start_selected_camera)
        btn_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止摄像头")
        self.stop_btn.clicked.connect(self.stop_selected_camera)
        btn_layout.addWidget(self.stop_btn)
        camera_layout.addLayout(btn_layout)

        self.add_video_btn = QPushButton("添加视频文件")
        self.add_video_btn.clicked.connect(self.add_video_source)
        camera_layout.addWidget(self.add_video_btn)

        # 识别阈值控制
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

        layout.addWidget(camera_group)

        # 系统状态显示
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
    def start_selected_camera(self):
        """启动所选摄像头"""
        cam_id = self.camera_combo.currentData()
        if cam_id is None:
            return
        if self.camera_manager.start_camera(cam_id):
            self.status_label.setText(f"已启动摄像头 {cam_id}")

    def stop_selected_camera(self):
        """停止所选摄像头"""
        cam_id = self.camera_combo.currentData()
        if cam_id is None:
            return
        if self.camera_manager.stop_camera(cam_id):
            self.status_label.setText(f"已停止摄像头 {cam_id}")

    def update_threshold(self, value):
        """调整识别置信度阈值"""
        threshold = value / 100
        self.face_detector.recognition_threshold = threshold
        self.threshold_value.setText(f"{threshold:.2f}")

    # ------------------------------
    # 主循环与图像处理
    # ------------------------------
    def update(self):
        """主循环（每30ms执行一次）：获取帧→识别→模糊→显示→更新状态"""
        try:
            frames = self.camera_manager.get_all_frames()
            total_faces = 0
            total_blurred = 0

            for cam_id, frame in frames.items():
                if frame is None:
                    continue

                if cam_id not in self.camera_labels:
                    self.add_camera_display(cam_id)
                    self.refresh_camera_combo()

                processed_frame, face_count, blurred_count = self.process_frame(cam_id, frame)
                total_faces += face_count
                total_blurred += blurred_count
                self.display_frame(cam_id, processed_frame)

            self.current_face_count = total_faces
            self.current_blurred_count = total_blurred
            self.status_label.setText(
                f"检测到人脸: {total_faces} | 已模糊: {total_blurred}"
            )
            self.update_status()

        except Exception as e:
            logger.error(f"更新循环错误: {e}")
            self.status_label.setText(f"错误: {str(e)}")

    def process_frame(self, cam_id: int, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """检测并处理一帧画面，返回处理结果及统计信息"""
        processed_frame = frame.copy()

        try:
            faces = self.face_detector.detect_faces(frame)
        except Exception as e:
            logger.error(f"摄像头 {cam_id} 检测人脸失败: {e}")
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
        if kernel % 2 == 0:
            kernel += 1
        return kernel

    def display_frame(self, cam_id: int, frame: np.ndarray):
        """将处理后的画面显示到对应摄像头窗口"""
        try:
            if frame is None:
                return
            pixmap = numpy_to_pixmap(frame)
            if pixmap is None:
                return
            if cam_id not in self.camera_labels:
                self.add_camera_display(cam_id)
            target_label = self.camera_labels.get(cam_id)
            if target_label is None:
                return
            scaled = pixmap.scaled(
                target_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            target_label.setPixmap(scaled)
        except Exception as e:
            logger.error(f"显示帧错误: {e}")

    def update_status(self):
        """更新系统状态信息：摄像头、人脸库、模糊统计"""
        try:
            status_text = []

            # 摄像头状态
            status_text.append("=== 摄像头状态 ===")
            for cam_id, cam_config in self.camera_manager.cameras.items():
                running = cam_id in self.camera_manager.capture_threads
                status_text.append(
                    f"摄像头 {cam_id}（{cam_config.name}）：{'运行中' if running else '已停止'}"
                )

            # 已知人脸状态
            status_text.append("\n=== 人脸库 ===")
            status_text.append(f"已知人脸数量：{len(self.face_detector.known_faces)}")

            # 实时模糊统计
            status_text.append("\n=== 实时统计 ===")
            status_text.append(f"当前检测到的人脸数量：{self.current_face_count}")
            status_text.append(f"当前被模糊的人脸数量：{self.current_blurred_count}")

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
                background-color: #1f2233;
                color: #f4f5f7;
            }
            QLabel {
                color: #f4f5f7;
            }
            QLabel#sectionTitle {
                font-size: 20px;
                font-weight: 600;
                padding: 8px 0;
            }
            QLabel#cameraTitle {
                font-size: 16px;
                font-weight: 600;
            }
            QLabel#cameraFeed {
                background-color: #151724;
                border-radius: 8px;
            }
            QTabWidget::pane {
                border: 1px solid #2b2f44;
                border-radius: 6px;
            }
            QTabBar::tab {
                padding: 10px 20px;
                background-color: transparent;
                color: #d0d3dc;
            }
            QTabBar::tab:selected {
                background-color: #2b2f44;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3a3f5c;
                border-radius: 6px;
                padding: 8px 16px;
                color: #ffffff;
            }
            QPushButton:hover {
                background-color: #50567a;
            }
            QPushButton:pressed {
                background-color: #2d324c;
            }
            QGroupBox {
                border: 1px solid #2b2f44;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
            }
            QScrollArea {
                border: none;
            }
            QFrame#cameraCard {
                background-color: #24283d;
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.05);
            }
        """)
