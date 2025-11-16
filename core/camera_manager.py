import cv2
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from loguru import logger
import time
import threading
import queue
import yaml
from pathlib import Path


@dataclass
class CameraConfig:
    id: int
    name: str
    source: str
    enabled: bool
    width: int
    height: int
    fps: int
    rotate: int


class CameraManager:
    """摄像头管理类：针对单摄像头的启动、捕获与状态管理"""

    def __init__(self, config_path: str):
        self.camera: Optional[CameraConfig] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        self.load_config(config_path)

    def _cleanup_camera_thread(self) -> None:
        """停止捕获线程并清空缓冲"""
        if self.capture_thread is None:
            return

        self.stop_event.set()
        self.capture_thread.join(timeout=2.0)
        if self.capture_thread.is_alive():
            logger.warning("摄像头线程未正常退出")

        try:
            while True:
                self.frame_queue.get_nowait()
        except queue.Empty:
            pass

        self.capture_thread = None
        logger.debug("已清理摄像头资源")

    def load_config(self, config_path: str) -> None:
        """从 YAML 文件加载单个摄像头配置"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}

            camera_cfg = config.get('camera')
            if not camera_cfg:
                raise ValueError("摄像头配置缺失")

            self.camera = CameraConfig(
                id=camera_cfg.get('id', 0),
                name=camera_cfg.get('name', 'Camera 0'),
                source=camera_cfg['source'],
                enabled=camera_cfg.get('enabled', True),
                width=camera_cfg['resolution']['width'],
                height=camera_cfg['resolution']['height'],
                fps=camera_cfg.get('fps', 30),
                rotate=camera_cfg.get('rotate', 0),
            )
            logger.info("成功加载摄像头配置")
        except Exception as e:
            logger.error(f"加载摄像头配置失败: {e}")
            raise

    def start_camera(self) -> bool:
        """启动摄像头"""
        if self.camera is None:
            logger.error("未加载摄像头配置")
            return False

        if not self.camera.enabled:
            logger.warning("摄像头在配置中被禁用")
            return False

        if self.capture_thread is not None and self.capture_thread.is_alive():
            logger.debug("摄像头已在运行")
            return True

        self.stop_event.clear()

        self.capture_thread = threading.Thread(
            target=self._capture_frames,
            daemon=True,
            name="CameraThread",
        )
        self.capture_thread.start()
        logger.info("摄像头已启动")
        return True

    def stop_camera(self) -> None:
        """停止摄像头"""
        self._cleanup_camera_thread()
        logger.info("摄像头已停止")

    def _capture_frames(self) -> None:
        """摄像头捕获线程函数"""
        if self.camera is None:
            return

        cam_config = self.camera
        cap: Optional[cv2.VideoCapture] = None

        try:
            source = int(cam_config.source) if str(cam_config.source).isdigit() else cam_config.source
            cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                logger.error(f"无法打开摄像头，源：{cam_config.source}")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_config.height)
            cap.set(cv2.CAP_PROP_FPS, cam_config.fps)

            logger.info("摄像头打开成功")

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("摄像头读取失败")
                    source_path = cam_config.source
                    if isinstance(source_path, str) and Path(source_path).exists():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    time.sleep(1)
                    continue

                if cam_config.rotate == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif cam_config.rotate == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif cam_config.rotate == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(frame)

        except Exception as e:
            logger.error(f"摄像头捕获线程出错: {e}")
        finally:
            if cap is not None:
                cap.release()
            logger.info("摄像头捕获线程退出")

    def get_frame(self) -> Optional[np.ndarray]:
        """获取最新帧"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def get_camera_status(self) -> Dict[str, object]:
        """获取摄像头状态信息"""
        if self.camera is None:
            return {}

        running = self.capture_thread is not None and self.capture_thread.is_alive()
        return {
            'id': self.camera.id,
            'name': self.camera.name,
            'running': running,
            'frame_queue_size': self.frame_queue.qsize(),
            'enabled': self.camera.enabled,
        }
