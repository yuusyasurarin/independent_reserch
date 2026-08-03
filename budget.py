items = [
    {"name": "ELPグローバルシャッターUSB3.0カメラ（60fps・5MP）", "price": 38299},
    {"name": "トランセンド ポータブルSSD 256GB", "price": 14841},
    {"name": "UGREEN USB3.0 延長ケーブル A オス A メス 2m", "price": 1168},
    {"name": "K&F Concept カメラ三脚 2m 4段伸縮", "price": 8245},
]

subtotal = sum(item["price"] for item in items)
print(f"購入予定合計: {subtotal:,}円")
for item in items:
    print(f"- {item['name']}: {item['price']:,}円")
