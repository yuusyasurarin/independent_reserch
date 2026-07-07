# 自主研究 -バスケットボールのトラッキングと時系列情報を用いた自動得点判定システムの開発 
ここのページには成果や記録を残していこうと思う。
## 予算
**総予算:** 122,749円

| 目的 | 商品名 | 備考 | 購入サイト |
| :--- | :--- | :--- | :--- |
| **メイン計算機**<br>（推論処理・システム制御） | NVIDIA Jetson Orin Nano 8GB RAM 公式基本キット (256GB SSD搭載) | **【確定】** 約83,099円。<br>Ubuntu・基本環境セットアップ済みの即戦力キット。 | [Amazon](https://www.amazon.co.jp/Yahboom-%E5%85%AC%E5%BC%8F%E9%96%8B%E7%99%BA%E3%82%AD%E3%83%83%E3%83%88%E3%80%90%E6%AD%A3%E8%A6%8F%E4%BB%A3%E7%90%86%E5%BA%97%E3%80%91-SD%E3%82%AB%E3%83%BC%E3%83%89%E3%82%B9%E3%83%AD%E3%83%83%E3%83%88%E4%BB%98%E3%81%8D-Isaac%E3%80%81%E3%83%AD%E3%83%9C%E3%83%86%E3%82%A3%E3%82%AF%E3%82%B9%E5%90%91%E3%81%91-Official/dp/B0CMHQ271L?ref_=ast_bl_cpl_dp&th=1)|
| **動画撮影**<br>（データ収集・推論入力） | ELP 5MP 50fps Global Shutter USB Camera (2.8-12mm 4倍ズーム) | **【確定】** 28,708円。<br>グローバルシャッター搭載で高速なボールの歪みを防止。 | [Amazon](https://www.amazon.co.jp/%E3%82%B0%E3%83%AD%E3%83%BC%E3%83%90%E3%83%AB%E3%82%B7%E3%83%A3%E3%83%83%E3%82%BF%E3%83%BC-%E6%89%8B%E5%8B%95%E3%82%BA%E3%83%BC%E3%83%A0%E3%82%A6%E3%82%A7%E3%83%96%E3%82%AB%E3%83%A1%E3%83%A9-%E3%83%90%E3%83%AA%E3%83%95%E3%82%A9%E3%83%BC%E3%82%AB%E3%83%AB%E8%B6%85%E6%9C%9B%E9%81%A0pc%E3%82%AB%E3%83%A1%E3%83%A9-USB%E7%9B%A3%E8%A6%96camera-%E3%82%B0%E3%83%AD%E3%83%BC%E3%83%90%E3%83%AB%E3%82%B7%E3%83%A3%E3%83%83%E3%82%BF%E3%83%BC4%E5%80%8D%E3%82%BA%E3%83%BC%E3%83%A0/dp/B0FRMBQFW2) |
| **データ保存**<br>（録画・推論結果・移動） | 高速microSDカード（256GBクラス / V30対応） | **【追加推奨】** 約4,000円。<br>USB帯域の競合を避け、システムを安定させるために必須。 | [Amazon](https://www.amazon.co.jp/KIOXIA-%E3%82%AD%E3%82%AA%E3%82%AF%E3%82%B7%E3%82%A2-microSDXC%E3%82%AB%E3%83%BC%E3%83%89-Amazon-co-jp%E3%83%A2%E3%83%87%E3%83%AB-KLMEA128G/dp/B08PTPTMH5/ref=sr_1_5?crid=4WUW5IXTJEFU&dib=eyJ2IjoiMSJ9.zyPa1-F_TmXJju6sibX96UaO9eOYekG23hvZAfDTXrvOrincAhbxQRWDTq8huACRqCQKBDkVRvUzpYhy492FIKZ2ktlvJ9a_wPkesuUSkCzC9lTya0O7DuJnpyXJ8EdJMgv34MNFT5fvBrWlkk4DdSMa4eWVWzzt4BYUCarcskegFqkaLDtXKbOJ0afTrwpMxB3FXJ0ifpmL_DPD9oqcq-zkI3bDifUy6y-K6lyCxQidqQGAFtqYhsodu7R_k7OuiGjSMKj5b7SucTnbiWNegb3JL6ewuvtIPBgW1DzXLv8._ztjz2e2-5I5VvKtuKykCdFuJ_f9YOG-cktyOK8TM90&dib_tag=se&keywords=%E3%83%9E%E3%82%A4%E3%82%AF%E3%83%ADsd%E3%82%AB%E3%83%BC%E3%83%89&qid=1782637771&sprefix=%E3%83%9E%E3%82%A4%E3%82%AF%E3%83%AD%2Caps%2C313&sr=8-5&th=1) |
| **カメラ固定** | 三脚 | **【既存活用】** <br>研究室の備品を使用するため予算計上なし。 | なし |
| **Jetson操作・画面確認** | 軽量モバイルモニター | **【個人所有】** <br>体育館での現地セットアップ用。個人所有物を使用。 | なし |
| **Jetson給電**<br>（体育館での完全ワイヤレス運用） | ① 65W PD対応モバイルバッテリー<br>② Type-C to DC トリガーケーブル (20V) | **【一部追加推奨】** <br>①は個人購入・所有物を使用。<br>②の変換ケーブル（約1,500円）のみ追加購入推奨。 | [Amazon](https://www.amazon.co.jp/SZSL-PD%E3%83%88%E3%83%AA%E3%82%AC%E3%83%BC%E3%82%B1%E3%83%BC%E3%83%96%E3%83%AB-PVC%E3%81%AE%E7%B4%A0%E6%9D%90%E3%82%92%E6%8E%A1%E7%94%A8-20V-1m/dp/B0D3T1RYCH/ref=sr_1_7?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&crid=23T42CCBZLOR&dib=eyJ2IjoiMSJ9.728D88BNXnL9VLoxBy_WuAdZAcMBhImRiG8D3tHSpgJZKSO5y0VJ9jgg2fhpPLttpWv7BFfEngdTj74eytuprbxsXc23h0PsTQVoBtSE9DRU4N-j7SwMjf6UiRQOIakKwIH-DC-4_W2olTI33FKKfVznXiU2c8XfdDYApFzwhnE6ytbwiK0cUfTXOGkPOh_UcyiHpJ0cM11T_fqtjx-I8mhP-4JcJ2ea61-ahSlgTy4tlUwYodzOsrAVPnxB_HJzFUDYcUgm3WJwEq8VDw7SYCxUEIRanh52ChZIXOlKZKw.HdfHF2JiIi-IsX9FvZXoLUMPr8QUQYpXpu-4s-A-EhQ&dib_tag=se&keywords=Type-C%2Bto%2BDC%2B20V%2B%E3%83%88%E3%83%AA%E3%82%AC%E3%83%BC%E3%82%B1%E3%83%BC%E3%83%96%E3%83%AB&qid=1782637139&sprefix=type-c%2Bto%2Bdc%2B20v%2B%E3%83%88%E3%83%AA%E3%82%AC%E3%83%BC%E3%82%B1%E3%83%BC%E3%83%96%E3%83%AB%2Caps%2C284&sr=8-7&ufe=app_do%3Aamzn1.fos.d8e7ee72-073f-4b97-8ec0-59c18d6dfebe&th=1)|
 
## 研究の予定
- 研究の詳細化をする。7月3週目まで
- 使用カメラをもらう
- 論文検索
- 
## 成果
## 記録
#### 5/19　
リポジトリの作成
#### 7/7　
堀井先生と面談をした。研究予定の詳細化をさらに進めることを課題にされた。目標ではできたかできなかったかの判定ができる内容にする。例えばシュートの認識精度が85%以上、トラッキングの精度が70％以上続くなど。論文と比較しながら決める。外部のデータセットがないかの確認をする。画角などを決める。最終的にもなぜこの画角にしたのか聞かれるかもしれない。前提条件を確認する。前提条件を確認して、なぜ自分の研究が必要でこのアプローチが有用なのかということを確認する。予定は1カ月ごとくらいには立てる。その際に「～を確認する」など曖昧なことではなくて、定量的に評価できる予定にする。タスク形にするとよいかもしれない。  

あと、研究に使用するPCをjetsonではなくて自分のwindowsPCにする方がいいのでは？と提案された。確かに、モニターを持ち運んだりjetsonの電源のことを考える必要がなくなるのでそれはいいかもしれない。
