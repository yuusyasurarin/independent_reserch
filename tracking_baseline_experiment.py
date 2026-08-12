"""OpenCV で既存のトラッキング手法を比較するベースライン実験スクリプト

使い方:
    python tracking_baseline_experiment.py --video sample.mp4 --tracker csrt
    python tracking_baseline_experiment.py --video sample.mp4 --tracker kcf --output results.csv
"""

import argparse
import csv
import os
from typing import List, Tuple

import cv2


TRACKER_CHOICES = [
    "csrt",
    "kcf",
    "mosse",
    "medianflow",
    "boosting",
    "mil",
    "tld",
    "goturn",
]


SELECTED_BOX = None
DRAWING = False
START_POINT = None
END_POINT = None


def on_mouse(event, x, y, flags, param):
    global SELECTED_BOX, DRAWING, START_POINT, END_POINT

    if event == cv2.EVENT_LBUTTONDOWN:
        DRAWING = True
        START_POINT = (x, y)
        END_POINT = (x, y)
        SELECTED_BOX = None
    elif event == cv2.EVENT_MOUSEMOVE:
        if DRAWING:
            END_POINT = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if START_POINT is not None and END_POINT is not None:
            x1 = min(START_POINT[0], END_POINT[0])
            y1 = min(START_POINT[1], END_POINT[1])
            x2 = max(START_POINT[0], END_POINT[0])
            y2 = max(START_POINT[1], END_POINT[1])
            SELECTED_BOX = (x1, y1, x2 - x1, y2 - y1)
        DRAWING = False


def select_roi(frame):
    """ユーザーが ROI をドラッグして選ぶ"""
    clone = frame.copy()
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select ROI", on_mouse)

    print("[Step 1] ボールが含まれる領域をドラッグして選択してください。Enterで確定、qで終了")

    while True:
        display = clone.copy()
        if DRAWING and START_POINT is not None and END_POINT is not None:
            cv2.rectangle(display, START_POINT, END_POINT, (0, 255, 0), 2)
        if SELECTED_BOX is not None:
            x, y, w, h = SELECTED_BOX
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Select ROI", display)
        key = cv2.waitKey(10) & 0xFF

        if key == 13:  # Enter
            if SELECTED_BOX is not None:
                cv2.destroyWindow("Select ROI")
                return tuple(int(v) for v in SELECTED_BOX)
            print("ROIが選択されていません。もう一度選択してください。")
        elif key == ord('q'):
            cv2.destroyWindow("Select ROI")
            raise KeyboardInterrupt

        if SELECTED_BOX is not None:
            clone = frame.copy()


def build_tracker(name: str):
    if name == "csrt":
        return cv2.TrackerCSRT_create()
    if name == "kcf":
        return cv2.TrackerKCF_create()
    if name == "mosse":
        return cv2.TrackerMOSSE_create()
    if name == "medianflow":
        return cv2.TrackerMedianFlow_create()
    if name == "boosting":
        return cv2.legacy.TrackerBoosting_create()
    if name == "mil":
        return cv2.TrackerMIL_create()
    if name == "tld":
        return cv2.legacy.TrackerTLD_create()
    if name == "goturn":
        return cv2.TrackerGOTURN_create()
    raise ValueError(f"Unsupported tracker: {name}")


def draw_box(frame, box, label):
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    center_x = int(x + w / 2)
    center_y = int(y + h / 2)
    cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
    cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def validate_box(box):
    x, y, w, h = [float(v) for v in box]
    if w <= 0 or h <= 0:
        return False
    return True


def run_tracker(video_path: str, tracker_name: str, output_csv: str = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("最初のフレームを読み取れませんでした。")

    # まず ROI を選択
    roi = select_roi(first_frame)
    tracker = build_tracker(tracker_name)
    ok = tracker.init(first_frame, roi)
    if not ok:
        raise RuntimeError(f"トラッカーの初期化に失敗しました: {tracker_name}")

    output_rows: List[Tuple[int, int, int]] = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ok, box = tracker.update(frame)
        frame_index += 1

        if ok and validate_box(box):
            x, y, w, h = [float(v) for v in box]
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            output_rows.append((frame_index, center_x, center_y))
            draw_box(frame, box, tracker_name.upper())
        else:
            cv2.putText(frame, "TRACK LOST", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("Tracking Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            save_path = os.path.join("captures", f"{tracker_name}_{frame_index}.png")
            os.makedirs("captures", exist_ok=True)
            cv2.imwrite(save_path, frame)
            print(f"保存しました: {save_path}")

    cap.release()
    cv2.destroyAllWindows()

    if output_csv:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "center_x", "center_y"])
            writer.writerows(output_rows)
        print(f"CSVを保存しました: {output_csv}")

    print(f"追跡結果のサンプル数: {len(output_rows)}")
    if output_rows:
        print(f"最初の結果: frame={output_rows[0][0]}, center=({output_rows[0][1]}, {output_rows[0][2]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="既存の OpenCV トラッキング手法を比較するベースライン実験")
    parser.add_argument("--video", type=str, required=True, help="実験用動画ファイルのパス")
    parser.add_argument("--tracker", type=str, default="csrt", choices=TRACKER_CHOICES, help="使用するトラッカー")
    parser.add_argument("--output", type=str, default="tracking_result.csv", help="追跡結果を保存するCSVファイル")
    args = parser.parse_args()

    run_tracker(args.video, args.tracker, args.output)
