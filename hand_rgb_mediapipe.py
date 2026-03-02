import time
from collections import Counter, deque
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


def vec_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cosv = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


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
        # z 方向缩放因子，用于把“相对深度”放大到一个可读的量级
        self.depth_scale = 1000.0
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
        """估一个稳定的 z 缩放系数（只用于相对深度，不追求真毫米）。"""
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
            if abs(avg_z) < 1e-4:
                return

            # 粗略利用 avg_z 估计一个缩放因子，并做限幅 + EMA，避免尺度动辄上万
            raw_scale = self.filter_cfg.expected_palm_width_mm / abs(avg_z)
            raw_scale = float(np.clip(raw_scale, 500.0, 5000.0))

            if not self.calibrated:
                self.depth_scale = raw_scale
                self.calibrated = True
            else:
                alpha_s = 0.1
                self.depth_scale = (1.0 - alpha_s) * self.depth_scale + alpha_s * raw_scale

    def process(self, raw_landmarks: np.ndarray, ts: float) -> dict:
        """入口：输入 (21,3) 归一化 + 相对 z，输出平滑+约束+相对深度坐标。"""
        lm = self._ema_filter(raw_landmarks, ts)
        lm = self._apply_bone_constraints(lm)
        self._update_calibration(lm)

        palm_w_norm = palm_width_from_landmarks(lm)
        if palm_w_norm < 1e-4:
            plane_scale = 1.0
        else:
            plane_scale = self.filter_cfg.expected_palm_width_mm / palm_w_norm

        lm_mm = lm.copy()
        # xy：按掌宽缩放到近似 mm 尺度
        lm_mm[:, 0:2] *= plane_scale
        # z：以平均 z 为中心的“相对深度”再乘一个稳定的缩放因子，只用于比较远近
        z_center = float(np.mean(lm[:, 2]))
        lm_mm[:, 2] = (lm[:, 2] - z_center) * self.depth_scale

        return {
            "lm_smooth": lm,
            "lm_mm": lm_mm,
            "depth_scale": self.depth_scale,
            "calibrated": self.calibrated,
        }


@dataclass
class SingleHandGesture:
    pinch: bool
    open_palm: bool
    fist: bool
    point: bool
    confidence: float


class GestureDetector:
    """基于 lm_mm 的简单单手手势识别：Pinch / OpenPalm / Fist / Point。"""

    def __init__(
        self,
        pinch_thresh_mm: float = 25.0,
        open_thresh_mm: float = 70.0,
        fist_thresh_mm: float = 35.0,
    ):
        self.pinch_thresh_mm = pinch_thresh_mm
        self.open_thresh_mm = open_thresh_mm
        self.fist_thresh_mm = fist_thresh_mm

    @staticmethod
    def _palm_center(lm_mm: np.ndarray) -> np.ndarray:
        palm_indices = [0, 5, 9, 13, 17]
        return np.mean(lm_mm[palm_indices, :3], axis=0)

    def detect_single(self, lm_mm: np.ndarray) -> SingleHandGesture:
        """输入一只手的 21 点 mm 坐标，输出基本手势（基于屈伸角 + Pinch 距离）。"""
        if lm_mm.shape[0] < 21:
            return SingleHandGesture(False, False, False, False, 0.0)

        # 各手指在 PIP 处的屈伸角
        fingers = {
            "index": (5, 6, 8),
            "middle": (9, 10, 12),
            "ring": (13, 14, 16),
            "pinky": (17, 18, 20),
        }
        angles = {}
        for name, (mcp, pip, tip) in fingers.items():
            v1 = lm_mm[mcp, :3] - lm_mm[pip, :3]
            v2 = lm_mm[tip, :3] - lm_mm[pip, :3]
            angles[name] = vec_angle_deg(v1, v2)

        extended = {k: v > 150.0 for k, v in angles.items()}
        flexed = {k: v < 60.0 for k, v in angles.items()}

        # Open：四指伸直
        is_open = all(extended.values())
        # Fist：四指弯曲
        is_fist = all(flexed.values())
        # Point：食指伸直，其他弯曲
        is_point = extended["index"] and all(
            flexed[f] for f in ("middle", "ring", "pinky")
        )

        # Pinch：拇指与食指靠近，且食指大致伸直
        thumb_index_dist = float(np.linalg.norm(lm_mm[4, :3] - lm_mm[8, :3]))
        is_pinch = (
            thumb_index_dist < self.pinch_thresh_mm
            and extended["index"]
            and not is_fist
        )

        # 置信度：根据与阈值的距离估计
        conf = 0.0
        if is_pinch:
            conf = max(conf, (self.pinch_thresh_mm - thumb_index_dist) / self.pinch_thresh_mm)
        if is_open:
            min_ext = min(angles.values())
            conf = max(conf, (min_ext - 150.0) / 30.0)
        if is_fist:
            max_flex = max(angles.values())
            conf = max(conf, (60.0 - max_flex) / 60.0)
        if is_point:
            conf = max(conf, 0.5)

        conf = float(np.clip(conf, 0.0, 1.0))

        return SingleHandGesture(
            pinch=is_pinch,
            open_palm=is_open,
            fist=is_fist,
            point=is_point,
            confidence=conf,
        )


class HandIdentityTracker:
    """根据手腕 2D 轨迹 + handedness 进行轻量身份跟踪，稳定分配到固定的滤波器槽位。"""

    def __init__(self, filter_cfg: FilterConfig, cons_cfg: ConstraintConfig, max_slots: int = 2):
        self.slots = []
        for _ in range(max_slots):
            self.slots.append(
                {
                    "postproc": HandPostProcessor(filter_cfg, cons_cfg),
                    "last_wrist": None,
                    "label_history": deque(maxlen=10),
                    "stable_label": "Unknown",
                    "gesture_history": deque(maxlen=8),
                    "stable_gesture": "",
                }
            )

    def update_and_process(self, detections, t_now: float, gesture_detector: GestureDetector):
        """
        detections: list of {
          "lm2d": hand_lms_2d,
          "lm_world": np.ndarray (21,3),
          "wrist_xy": (x_pix, y_pix),
          "label": "Left"/"Right"/""
        }
        返回：每只跟踪手的稳定输出列表。
        """
        outputs = []
        used_slots = set()

        for det in detections:
            wx, wy = det["wrist_xy"]

            # 找到距离上一帧 wrist 最近的槽位
            best_idx = None
            best_dist = None
            for idx, slot in enumerate(self.slots):
                if idx in used_slots:
                    continue
                lw = slot["last_wrist"]
                if lw is None:
                    best_idx = idx
                    break
                dx = wx - lw[0]
                dy = wy - lw[1]
                dist = dx * dx + dy * dy
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is None:
                continue

            slot = self.slots[best_idx]
            used_slots.add(best_idx)
            slot["last_wrist"] = (wx, wy)

            # handedness 作为弱证据，通过多数投票稳定标签
            raw_label = det.get("label") or ""
            if raw_label:
                slot["label_history"].append(raw_label)
                if len(slot["label_history"]) >= 5:
                    counts = Counter(slot["label_history"])
                    slot["stable_label"] = max(counts, key=counts.get)
                else:
                    slot["stable_label"] = raw_label

            proc = slot["postproc"].process(det["lm_world"], t_now)
            lm_mm = proc["lm_mm"]
            gesture = gesture_detector.detect_single(lm_mm)

            # 手势名称 + 置信度门控
            gname = "None"
            if gesture.confidence >= 0.5:
                if gesture.pinch:
                    gname = "Pinch"
                elif gesture.fist:
                    gname = "Fist"
                elif gesture.open_palm:
                    gname = "Open"
                elif gesture.point:
                    gname = "Point"

            slot["gesture_history"].append(gname)

            # 多帧投票形成稳定手势，避免抖动
            stable = slot["stable_gesture"]
            if len(slot["gesture_history"]) >= 4:
                gc = Counter(slot["gesture_history"])
                candidate, count = gc.most_common(1)[0]
                if count >= 2:
                    slot["stable_gesture"] = candidate
                    stable = candidate
            else:
                slot["stable_gesture"] = gname
                stable = gname

            outputs.append(
                {
                    "label": slot["stable_label"],
                    "gesture_name": "" if stable == "None" else stable,
                    "gesture_conf": gesture.confidence,
                    "proc": proc,
                    "lm2d": det["lm2d"],
                }
            )

        return outputs


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
    gesture_detector = GestureDetector()
    identity_tracker = HandIdentityTracker(filter_cfg, cons_cfg, max_slots=2)
    last_gesture_name = {}

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
            detections = []
            for hand_idx, (hand_lms_2d, hand_lms_3d) in enumerate(
                zip(result.hand_landmarks, result.hand_world_landmarks)
            ):
                raw_label = ""
                if result.handedness and hand_idx < len(result.handedness):
                    cat = result.handedness[hand_idx][0]
                    raw_label = cat.category_name

                wrist_2d = hand_lms_2d[0]
                wrist_xy = (wrist_2d.x * w, wrist_2d.y * h)
                lm_world = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_lms_3d],
                    dtype=np.float32,
                )
                detections.append(
                    {
                        "lm2d": hand_lms_2d,
                        "lm_world": lm_world,
                        "wrist_xy": wrist_xy,
                        "label": raw_label,
                    }
                )

            tracked = identity_tracker.update_and_process(
                detections, t_now, gesture_detector
            )

            for item in tracked:
                hand_label = item["label"] or "Unknown"
                proc = item["proc"]
                lm_mm = proc["lm_mm"]
                gesture_name = item["gesture_name"]

                # 画 2D 关键点，不同颜色区分左右
                if hand_label == "Left":
                    color = (255, 0, 0)
                    label_short = "L"
                    y_offset = 30
                elif hand_label == "Right":
                    color = (0, 255, 0)
                    label_short = "R"
                    y_offset = 60
                else:
                    color = (0, 255, 255)
                    label_short = "U"
                    y_offset = 90

                for lm_2d in item["lm2d"]:
                    x_pix = int(lm_2d.x * w)
                    y_pix = int(lm_2d.y * h)
                    cv2.circle(frame, (x_pix, y_pix), 3, color, -1)

                text = f"{label_short}_z:{lm_mm[0, 2]:.0f} {gesture_name}"
                cv2.putText(
                    frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

                # 仅在稳定手势变化时打印一条控制台日志
                key = hand_label or f"slot-{id(item)}"
                if gesture_name != last_gesture_name.get(key, ""):
                    last_gesture_name[key] = gesture_name
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"{hand_label or 'Unknown'}: z={lm_mm[0, 2]:.1f}mm "
                        f"gesture={gesture_name or 'None'} "
                        f"conf={item['gesture_conf']:.2f} "
                        f"depth_scale={proc['depth_scale']:.2f} "
                        f"calib={proc['calibrated']}"
                    )

        cv2.imshow("Mono RGB + MediaPipe Hands 3D", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_demo()

