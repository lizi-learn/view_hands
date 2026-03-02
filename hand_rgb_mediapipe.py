import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


@dataclass
class FilterConfig:
    ema_alpha: float = 0.4          # EMA 平滑系数，0.3~0.6 比较合适
    hold_last_seconds: float = 0.15  # 丢失跟踪后保留上一帧的时间
    min_calib_frames: int = 50       # 标定需要的最少帧数
    expected_palm_width_mm: float = 80.0  # 预期手掌宽度（mm，粗略）


@dataclass
class ConstraintConfig:
    max_bone_scale_change: float = 1.6
    min_bone_scale_change: float = 0.5


def palm_width_from_landmarks(landmarks3d: np.ndarray) -> float:
    """利用食指 MCP(5) 到小指 MCP(17) 的距离近似手掌宽度（归一化单位）"""
    p1 = landmarks3d[5]
    p2 = landmarks3d[17]
    return float(np.linalg.norm(p1 - p2))


BONE_PAIRS = [
    (0, 5), (5, 6), (6, 7), (7, 8),          # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),     # 中指
    (0, 13), (13, 14), (14, 15), (15, 16),   # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20),   # 小指
]


class HandPostProcessor:
    """对 MediaPipe Hands 的 21 点 3D 结果做：EMA 滤波 + 骨长约束 + 简单深度标定。"""

    def __init__(self, filter_cfg: FilterConfig, cons_cfg: ConstraintConfig):
        self.filter_cfg = filter_cfg
        self.cons_cfg = cons_cfg

        self.last_landmarks = None
        self.last_timestamp = None
        self.bone_lengths_ref = None

        self.calib_palm_widths = deque()
        self.calib_z_values = deque()
        self.depth_scale = 1.0
        self.calibrated = False

    def _ema_filter(self, new_lm: np.ndarray, ts: float) -> np.ndarray:
        """简单指数平滑滤波"""
        alpha = self.filter_cfg.ema_alpha
        if self.last_landmarks is None:
            self.last_landmarks = new_lm
            self.last_timestamp = ts
            return new_lm

        if ts - self.last_timestamp > self.filter_cfg.hold_last_seconds * 3:
            # 如果长时间没有新数据，直接重置
            self.last_landmarks = new_lm
            self.last_timestamp = ts
            return new_lm

        filt = alpha * new_lm + (1.0 - alpha) * self.last_landmarks
        self.last_landmarks = filt
        self.last_timestamp = ts
        return filt

    def _apply_bone_constraints(self, lm: np.ndarray) -> np.ndarray:
        """利用骨骼长度变化范围做一个极简约束，防止某帧崩坏。"""
        pts = lm.copy()
        n_bones = len(BONE_PAIRS)
        bone_lengths = np.zeros(n_bones, dtype=np.float32)

        for i, (a, b) in enumerate(BONE_PAIRS):
            bone_lengths[i] = np.linalg.norm(pts[a] - pts[b])

        if self.bone_lengths_ref is None:
            self.bone_lengths_ref = bone_lengths
            return pts

        max_scale = self.cons_cfg.max_bone_scale_change
        min_scale = self.cons_cfg.min_bone_scale_change

        for i, (a, b) in enumerate(BONE_PAIRS):
            ref_len = self.bone_lengths_ref[i]
            cur_len = bone_lengths[i]
            if ref_len < 1e-6 or cur_len < 1e-6:
                continue

            scale = cur_len / ref_len
            if scale > max_scale or scale < min_scale:
                clamped_scale = float(np.clip(scale, min_scale, max_scale))
                target_len = clamped_scale * ref_len

                dir_vec = pts[b] - pts[a]
                norm = float(np.linalg.norm(dir_vec))
                if norm < 1e-6:
                    continue
                dir_unit = dir_vec / norm
                pts[b] = pts[a] + dir_unit * target_len

        return pts

    def _update_calibration(self, lm: np.ndarray) -> None:
        """简单线性标定：假设 depth_mm ≈ s * z，相机固定后估一个 s。"""
        palm_w = palm_width_from_landmarks(lm)
        mean_z = float(np.mean(lm[:, 2]))
        if palm_w < 1e-4 or abs(mean_z) < 1e-6:
            return

        self.calib_palm_widths.append(palm_w)
        self.calib_z_values.append(mean_z)

        if len(self.calib_palm_widths) > self.filter_cfg.min_calib_frames:
            while len(self.calib_palm_widths) > self.filter_cfg.min_calib_frames:
                self.calib_palm_widths.popleft()
                self.calib_z_values.popleft()

            avg_z = float(np.mean(self.calib_z_values))
            self.depth_scale = self.filter_cfg.expected_palm_width_mm / abs(avg_z)
            self.calibrated = True

    def process(self, raw_landmarks: np.ndarray, ts: float) -> dict:
        """入口：输入 (21,3) 归一化 + 相对 z，输出平滑+约束+近似 mm 坐标。"""
        lm = self._ema_filter(raw_landmarks, ts)
        lm = self._apply_bone_constraints(lm)
        self._update_calibration(lm)

        palm_w_norm = palm_width_from_landmarks(lm)
        if palm_w_norm < 1e-4:
            plane_scale = 1.0
        else:
            plane_scale = self.filter_cfg.expected_palm_width_mm / palm_w_norm

        lm_mm = lm.copy()
        lm_mm[:, 0:2] *= plane_scale
        lm_mm[:, 2] *= self.depth_scale

        return {
            "lm_smooth": lm,
            "lm_mm": lm_mm,
            "depth_scale": self.depth_scale,
            "calibrated": self.calibrated,
        }


def run_camera_demo():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 使用 MediaPipe Tasks API 的 HandLandmarker
    # 需要本地 hand_landmarker.task 模型文件
    base_options = mp_python.BaseOptions(
        model_asset_path="/home/pc/test/hand_landmarker.task"
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,  # 支持双手
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)

    filter_cfg = FilterConfig()
    cons_cfg = ConstraintConfig()
    # 为左右手各维护一个独立的后处理器，避免索引交换导致抖动
    postprocs = {
        "Left": HandPostProcessor(filter_cfg, cons_cfg),
        "Right": HandPostProcessor(filter_cfg, cons_cfg),
    }

    print("按 'q' 退出；保持手掌正对摄像头几秒用于自动标定深度。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        t_now = time.time()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = hand_landmarker.detect(mp_image)

        if result.hand_landmarks and result.hand_world_landmarks:
            # 多手：逐只处理，并根据 handedness 区分左右手
            for hand_idx, (hand_lms_2d, hand_lms_3d) in enumerate(
                zip(result.hand_landmarks, result.hand_world_landmarks)
            ):
                # 读取 MediaPipe 提供的左右手分类（"Left" / "Right"）
                hand_label = "Unknown"
                if result.handedness and hand_idx < len(result.handedness):
                    cat = result.handedness[hand_idx][0]
                    hand_label = cat.category_name

                if hand_label not in postprocs:
                    continue

                lm_array = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lms_3d],
                    dtype=np.float32,
                )

                proc = postprocs[hand_label].process(lm_array, t_now)
                lm_mm = proc["lm_mm"]

                # 画 2D 关键点，不同颜色区分左右
                if hand_label == "Left":
                    color = (255, 0, 0)  # 左手：蓝色
                    label_short = "L"
                    y_offset = 30
                else:
                    color = (0, 255, 0)  # 右手：绿色
                    label_short = "R"
                    y_offset = 60

                for lm_2d in hand_lms_2d:
                    x_pix = int(lm_2d.x * w)
                    y_pix = int(lm_2d.y * h)
                    cv2.circle(frame, (x_pix, y_pix), 3, color, -1)

                # 左上角叠加每只手的手腕深度信息
                text = f"{label_short}_wrist_z_mm: {lm_mm[0, 2]:.1f}"
                cv2.putText(
                    frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        cv2.imshow("Mono RGB + MediaPipe Hands 3D", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_demo()

