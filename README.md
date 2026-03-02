## view_hands：单目 RGB + MediaPipe 3D 手势 Demo

一个基于 **Python 3.12 + MediaPipe Tasks + OpenCV** 的实时 3D 手部追踪与手势识别小工具，特点：

- **单摄像头 + CPU 可跑**，无深度相机依赖
- 使用 **MediaPipe HandLandmarker (Tasks API)** 获取 21 点 3D world landmarks
- 自带一层 **工程级后处理**：
  - 指数平滑 + 自适应门控
  - 骨长约束，避免关键点“崩坏”
  - 简单深度标定与限幅，输出“相对深度”
  - 左右手身份跟踪（Hand Identity Tracker），避免 Left/Right 互换导致抖动
- 内置基础手势识别与去抖：
  - 单手：Open / Fist / Point / Pinch
  - 双手：左右手独立识别
- 终端打印结构化日志，方便接入上层交互 / RPA / 多模态系统

---

## 环境准备

### Python

使用用户本地编译的 Python 3.12：

```bash
~/Python-3.12.2/python --version
```

项目目录下使用虚拟环境 `.venv`：

```bash
cd ~/test
~/Python-3.12.2/python -m venv .venv
source .venv/bin/activate
```

### 安装依赖

在 `.venv` 中安装：

```bash
pip install --upgrade pip
pip install mediapipe opencv-python numpy
```

确保 `hand_landmarker.task` 模型文件存在于当前目录（`~/test/hand_landmarker.task`）。

---

## 运行

```bash
cd ~/test
source .venv/bin/activate
python hand_rgb_mediapipe.py
```

- 窗口会显示摄像头画面与双手 2D 关键点（左手蓝色，右手绿色）
- 左上角文字：
  - `L_z:xxx <gesture>`：左手手腕相对深度 + 稳定手势
  - `R_z:xxx <gesture>`：右手同理
- 终端会在手势状态变化时输出一条日志，包含：
  - 时间、左右手、`z` 相对深度、手势名称、置信度、`depth_scale`、是否已标定

按 `q` 退出。

---

## 手势说明

当前支持的基础手势（单手）：

- **Open**：四指伸直、手掌展开
- **Fist**：四指弯曲，握拳
- **Point**：食指伸直，其它手指弯曲
- **Pinch**：拇指与食指捏合，且食指基本伸直

所有手势均基于 **3D 关节屈伸角 + 多帧投票去抖**，更适合做真实交互输入，而不是只做可视化 demo。

---

## 深度与标定说明

- MediaPipe 的 `world_landmarks` 不提供真实毫米深度，本项目输出的 `z` 是：
  - 相对于当前帧手掌平面的“前后”距离；
  - 乘以一个平滑的缩放因子 `depth_scale`（有限幅），只保证**远近可比较**，不保证绝对值是真实毫米。
- 若需要更精确的物理深度，需要额外的相机标定流程或深度传感器。

---

## 后续可扩展方向（建议）

- 更丰富的双手组合手势（缩放 / 平移 / 旋转）
- 导出统一 `HandState` / `GestureEvent` 流，接入语音、眼动等多模态系统
- 引入简易相机标定，使 z 更接近真实距离
