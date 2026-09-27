"""生成 FaceSwitch 应用图标: 渐变圆角方块 + FS 字样 → build/icon.ico"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = 512
img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# 渐变背景 (左上 #4f8cff → 右下 #8a63ff)
c1, c2 = (79, 140, 255), (138, 99, 255)
grad = Image.new("RGBA", (SIZE, SIZE))
gd = ImageDraw.Draw(grad)
for y in range(SIZE):
    for_step = y / SIZE
    r = int(c1[0] + (c2[0] - c1[0]) * for_step)
    g = int(c1[1] + (c2[1] - c1[1]) * for_step)
    b = int(c1[2] + (c2[2] - c1[2]) * for_step)
    gd.line([(0, y), (SIZE, y)], fill=(r, g, b, 255))

# 圆角遮罩
mask = Image.new("L", (SIZE, SIZE), 0)
md = ImageDraw.Draw(mask)
RADIUS = int(SIZE * 0.22)
md.rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], radius=RADIUS, fill=255)
img.paste(grad, (0, 0), mask)

# FS 文字
font = None
for cand in ("C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/msyhbd.ttc"):
    if Path(cand).exists():
        font = ImageFont.truetype(cand, int(SIZE * 0.42))
        break
if font is None:
    font = ImageFont.load_default()
text = "FS"
bbox = d.textbbox((0, 0), text, font=font)
w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
d.text(((SIZE - w) / 2 - bbox[0], (SIZE - h) / 2 - bbox[1]), text, font=font, fill=(255, 255, 255, 255))

out = Path("build")
out.mkdir(exist_ok=True)
img.save(out / "icon.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
img.save(out / "icon.png")
print("icon written:", out / "icon.ico")
