import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw
import os

from utils.const import BIG_IMAGE_HEIGHT, BIG_IMAGE_WIDTH


class LineDrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("line drawing")

        # キャンバスサイズ
        self.canvas_width = BIG_IMAGE_WIDTH
        self.canvas_height = BIG_IMAGE_HEIGHT

        # PILイメージを作成（グレースケールモード）
        self.image = Image.new(
            "L", (self.canvas_width, self.canvas_height), "black")
        self.draw = ImageDraw.Draw(self.image)

        # Tkinterキャンバス
        self.canvas = tk.Canvas(
            root, width=self.canvas_width, height=self.canvas_height, bg="black")
        self.canvas.pack()

        # マウスのイベント設定
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)

        # 保存ボタン
        self.save_button = tk.Button(root, text="保存", command=self.save_file)
        self.save_button.pack()

        # 描画する線の太さ
        self.line_width = 2
        self.previous_x = None
        self.previous_y = None

    def start_draw(self, event):
        self.previous_x = event.x
        self.previous_y = event.y

    def draw_line(self, event):
        if self.previous_x and self.previous_y:
            # キャンバスに線を描画
            self.canvas.create_line(
                self.previous_x, self.previous_y, event.x, event.y,
                fill="white", width=self.line_width
            )
            # PILイメージに線を描画
            self.draw.line(
                [self.previous_x, self.previous_y, event.x, event.y],
                fill="white", width=self.line_width
            )
        self.previous_x = event.x
        self.previous_y = event.y

    def save_file(self):
        # inputディレクトリに保存
        if not os.path.exists("input"):
            os.mkdir("input")

        idx = len(os.listdir("input"))
        file_path = f"input/{idx}.png"

        # PILイメージを保存
        self.image.save(file_path)
        print(f"保存しました: {file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LineDrawingApp(root)
    root.mainloop()
