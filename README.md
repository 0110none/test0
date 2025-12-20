# 单摄像头人脸识别与隐私保护系统 🔒

一款基于 **PyQt5** 与 **OpenCV** 的桌面应用，聚焦单路摄像头的人脸检测、识别与陌生人模糊处理。系统启动前提供本地账号密码登录，默认账号/密码均为 `123456`，支持在登录界面直接修改密码并写入 `config/user_config.json`。

## ✨ 核心功能
- **单路视频输入**：仅维护一个摄像头/视频源，实时采集与展示。
- **隐私保护**：已知人脸保持清晰，陌生人脸自适应高斯模糊。
- **人脸库管理**：内置对话框可导入、更新、删除已知人脸。
- **参数即时调节**：识别阈值、模糊强度在界面内滑动调整。
- **科技感 UI**：深色主题、渐变按钮、发光描边与圆角卡片式布局。

## 🗂️ 项目结构
```
.
├── assets/                # 资源文件（Logo 等）
├── config/
│   ├── camera_config.yaml # 单摄像头配置
│   ├── config.yaml        # 应用/模型/处理参数
│   └── user_config.json   # 本地登录账号密码
├── core/                  # 核心逻辑（摄像头、人脸检测等）
├── ui/
│   ├── face_manager.py    # 人脸库管理对话框
│   ├── login_window.py    # 登录与修改密码对话框
│   └── main_window.py     # 主界面（单摄像头监控 + 控制面板）
├── data/                  # 已知人脸库（自动创建）
├── logs/                  # 日志目录（自动创建）
├── main.py                # 程序入口，启动登录与主界面
└── requirements.txt       # 依赖列表
```

## ⚙️ 配置说明
### 应用与模型 (`config/config.yaml`)
- `app`: 应用名称、版本、Logo 路径、人脸库目录、日志目录。
- `recognition`: 检测/识别阈值、设备类型（cuda/cpu）、是否开启分析。
- `processing`: `blur_strength` 控制模糊范围倍率。

### 摄像头 (`config/camera_config.yaml`)
示例：
```yaml
camera:
  id: 0
  name: "Camera 0"
  source: 0          # 摄像头序号、RTSP 地址或视频文件
  enabled: true
  resolution:
    width: 640
    height: 480
  fps: 30
  rotate: 0
```

### 登录 (`config/user_config.json`)
默认内容：
```json
{"username": "123456", "password": "123456"}
```
可在登录界面通过“修改密码”按钮更新，并自动写回该文件。

## 🚀 使用步骤
1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
2. **准备目录**（首次运行自动创建）
   ```bash
   mkdir -p data/known_faces logs
   ```
3. **配置摄像头**：编辑 `config/camera_config.yaml` 指定单个源。
4. **启动程序**
   ```bash
   python main.py
   ```
5. **登录与修改密码**
   - 默认账号/密码均为 `123456`。
   - 登录框点击“修改密码”可验证旧密码并保存新密码。

## 🖥️ 主界面操作
- **监控区**：左侧卡片展示实时视频；已知人脸标注，陌生人自动模糊。
- **控制区**：右侧卡片可启动/停止摄像头、调整识别阈值与模糊强度。
- **状态面板**：显示摄像头运行状态、已知人脸数量、实时统计数据。

## 📄 毕业论文题目参考（人工智能方向）
如果你需要为人工智能相关的毕业设计撰写论文题目，可参考 `docs/thesis_topics.md` 中基于本项目特色整理的选题列表。

## 🛠️ 调试提示
- 日志默认写入 `logs/` 目录，便于排查摄像头或识别问题。
- 若摄像头源不可用，请确认 `source` 参数或分辨率设置与设备匹配。
