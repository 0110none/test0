# -*- coding: utf-8 -*-
"""登录与密码修改界面"""

import json
from pathlib import Path
from typing import Dict

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QWidget,
    QSpacerItem,
    QSizePolicy,
)

DEFAULT_USER = {"username": "123456", "password": "123456"}


def ensure_user_config(config_path: Path) -> Dict[str, str]:
    """确保用户配置文件存在并返回内容"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(json.dumps(DEFAULT_USER, indent=2), encoding="utf-8")
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        data.setdefault("username", DEFAULT_USER["username"])
        data.setdefault("password", DEFAULT_USER["password"])
        config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data
    except json.JSONDecodeError:
        config_path.write_text(json.dumps(DEFAULT_USER, indent=2), encoding="utf-8")
        return DEFAULT_USER.copy()


def save_user_config(config_path: Path, data: Dict[str, str]) -> None:
    """保存用户配置到本地"""
    config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


class PasswordDialog(QDialog):
    """修改密码对话框"""

    def __init__(self, config_path: Path, parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("修改密码")
        self.setModal(True)
        self.config_path = config_path
        self.user_data = ensure_user_config(self.config_path)
        self.setMinimumWidth(360)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        tip = QLabel("请验证旧密码后设置新密码")
        tip.setObjectName("dialogHint")
        layout.addWidget(tip)

        self.old_password = QLineEdit()
        self.old_password.setEchoMode(QLineEdit.Password)
        self.old_password.setPlaceholderText("旧密码")
        layout.addWidget(self.old_password)

        self.new_password = QLineEdit()
        self.new_password.setEchoMode(QLineEdit.Password)
        self.new_password.setPlaceholderText("新密码")
        layout.addWidget(self.new_password)

        self.confirm_password = QLineEdit()
        self.confirm_password.setEchoMode(QLineEdit.Password)
        self.confirm_password.setPlaceholderText("确认新密码")
        layout.addWidget(self.confirm_password)

        button_row = QHBoxLayout()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.save_new_password)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(save_btn)
        button_row.addWidget(cancel_btn)
        layout.addLayout(button_row)

    def save_new_password(self):
        """校验并保存新密码"""
        old = self.old_password.text().strip()
        new = self.new_password.text().strip()
        confirm = self.confirm_password.text().strip()

        if old != self.user_data.get("password"):
            QMessageBox.warning(self, "错误", "旧密码不正确")
            return

        if not new:
            QMessageBox.warning(self, "提示", "新密码不能为空")
            return

        if new != confirm:
            QMessageBox.warning(self, "提示", "两次输入的新密码不一致")
            return

        self.user_data["password"] = new
        save_user_config(self.config_path, self.user_data)
        QMessageBox.information(self, "成功", "密码已更新")
        self.accept()


class LoginWindow(QDialog):
    """登录对话框，验证本地用户名密码"""

    def __init__(self, config_path: Path, app_icon: str = "", parent: QWidget = None):
        super().__init__(parent)
        self.config_path = config_path
        self.user_data = ensure_user_config(self.config_path)
        self.setWindowTitle("登录系统")
        self.setWindowIcon(QIcon(app_icon) if app_icon else QIcon())
        self.setFixedSize(420, 360)
        self._build_ui()
        self.apply_styles()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("单摄像头人脸识别系统")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("dialogTitle")
        layout.addWidget(title)

        subtitle = QLabel("请登录后使用主界面")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setObjectName("dialogHint")
        layout.addWidget(subtitle)

        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("账号")
        self.username_edit.setText(self.user_data.get("username", ""))
        layout.addWidget(self.username_edit)

        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("密码")
        self.password_edit.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_edit)

        button_row = QHBoxLayout()
        self.login_btn = QPushButton("登录")
        self.login_btn.clicked.connect(self.handle_login)
        change_btn = QPushButton("修改密码")
        change_btn.clicked.connect(self.open_change_password)
        button_row.addWidget(self.login_btn)
        button_row.addWidget(change_btn)
        layout.addLayout(button_row)

        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.hint_label = QLabel("默认账号/密码：123456")
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setObjectName("dialogHint")
        layout.addWidget(self.hint_label)

    def handle_login(self):
        """校验登录信息"""
        username = self.username_edit.text().strip()
        password = self.password_edit.text().strip()
        if username == self.user_data.get("username") and password == self.user_data.get("password"):
            QMessageBox.information(self, "欢迎", "登录成功")
            self.accept()
        else:
            QMessageBox.warning(self, "失败", "账号或密码错误")

    def open_change_password(self):
        """打开修改密码对话框"""
        dialog = PasswordDialog(self.config_path, self)
        dialog.exec_()
        self.user_data = ensure_user_config(self.config_path)

    def apply_styles(self):
        """为对话框应用科技感暗色主题"""
        self.setStyleSheet(
            """
            QDialog {
                background-color: #0b1021;
                color: #e5e7eb;
                border: 1px solid #3b82f6;
                border-radius: 14px;
            }
            QLabel#dialogTitle {
                font-size: 20px;
                font-weight: 700;
                color: #7dd3fc;
                padding: 12px 0;
            }
            QLabel#dialogHint {
                color: #cbd5f5;
            }
            QLineEdit {
                padding: 12px 14px;
                border-radius: 10px;
                border: 1px solid #1d4ed8;
                background-color: rgba(255, 255, 255, 0.05);
                color: #e5e7eb;
            }
            QLineEdit:focus {
                border: 1px solid #60a5fa;
                box-shadow: 0 0 8px rgba(96,165,250,0.6);
            }
            QPushButton {
                padding: 12px 16px;
                border-radius: 12px;
                border: 1px solid #3b82f6;
                color: #e5e7eb;
                font-weight: 600;
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #1e3a8a, stop:1 #0ea5e9);
            }
            QPushButton:hover {
                border: 1px solid #7dd3fc;
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #2563eb, stop:1 #06b6d4);
            }
            QPushButton:pressed {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #0f172a, stop:1 #1e293b);
            }
        """
        )
