# 出典・固定バージョン

| 用途 | 実装・固定版 | ライセンス・出典 |
|---|---|---|
| 従来YOLOv5 | Ultralytics YOLOv5 v7.0、commit `915bbf294bb74c859f0b41f1c23bc395014ea679` | [固定ソース](https://github.com/ultralytics/yolov5/tree/915bbf294bb74c859f0b41f1c23bc395014ea679)、このcommitのLICENSEはGPL-3.0 |
| 初期重み | 公式v7.0 `yolov5s.pt`／`yolov5m.pt` | [公式リリース](https://github.com/ultralytics/yolov5/releases/tag/v7.0) |
| ByteTrack | Roboflow Supervision 0.25.0 | [固定版実装](https://github.com/roboflow/supervision/blob/0.25.0/supervision/tracker/byte_tracker/core.py)、[MIT License](https://github.com/roboflow/supervision/blob/0.25.0/LICENSE.md) |
| ByteTrackの元研究 | ByteTrack: Multi-Object Tracking by Associating Every Detection Box | [論文](https://arxiv.org/abs/2110.06864)、[公式実装](https://github.com/ifzhang/ByteTrack) |
| KCF | opencv-contrib-python 4.10.0.84のTrackerKCF | [OpenCV contrib 4.10.0 tracking](https://github.com/opencv/opencv_contrib/tree/4.10.0/modules/tracking)、Apache-2.0 |
| 学習・推論ランタイム | PyTorch 2.5.1／torchvision 0.20.1、CUDA 12.1 wheel | [公式旧版インストール手順](https://pytorch.org/get-started/previous-versions/#v251) |

YOLOv5ソースはsetup時に固定commitから取得し、元LICENSEをvendor内に保持します。
`vendor/yolov5/midterm_source.json` にソースZIPと各Python/YAMLファイルのSHA-256を保存し、実行前に照合します。
モデルごとに公式取得URL・SHA-256または学習run情報を記録します。
Python 3.12対応のPillow 10で削除された `font.getsize` は、`src/compat.py` の明示的な描画互換処理で補います。
vendor内のソースやモデル計算は変更しません。学習metadataに互換処理と自作wrapperのハッシュを記録します。
SupervisionのByteTrackを検出器から独立して利用し、Ultralyticsの現行YOLOモデルAPIは呼びません。
このフォルダの学習・推論wrapperと配布時の扱いは、同梱・依存する上記ライセンス条件に従ってください。
