"""カメラで撮影しながらYOLO + ByteTrackで追跡し、映像とCSVを保存する。"""

from yolo_basketball_tracker import parse_args, run


if __name__ == "__main__":
    try:
        run(parse_args("live"))
    except KeyboardInterrupt:
        print("終了しました。")
    except (RuntimeError, OSError) as error:
        raise SystemExit(f"エラー: {error}") from error
