"""録画動画の各フレームをYOLO + ByteTrackで処理し、結果動画とCSVを保存する。"""

from yolo_basketball_tracker import parse_args, run


if __name__ == "__main__":
    try:
        run(parse_args("video"))
    except KeyboardInterrupt:
        print("終了しました。")
    except (RuntimeError, OSError) as error:
        raise SystemExit(f"エラー: {error}") from error
