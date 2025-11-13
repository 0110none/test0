import numpy as np
from loguru import logger


def numpy_to_pixmap(image: np.ndarray) -> "QPixmap":
    """将 numpy 图像数据转换为 QPixmap（用于 PyQt 显示）"""
    try:
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtCore import Qt

        if image is None:
            return QPixmap()

        if len(image.shape) == 2:  # 灰度图
            h, w = image.shape
            qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        else:  # 彩色图（BGR 格式）
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)

        # 转换为 QPixmap，并保持比例缩放
        return QPixmap.fromImage(qimg).scaled(
            w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    except Exception as e:
        logger.error(f"numpy 转 QPixmap 出错: {e}")
        return QPixmap()
