import tensorflow as tf
from PIL import Image, ImageTk
import tkinter as tk
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# Function to generate Mandelbrot set
def mandelbrot_tf(c, max_iter):
    z = c
    not_diverged = tf.abs(z) <= 2
    for _ in range(max_iter):
        z = tf.where(not_diverged, z ** 2 + c, z)
        not_diverged = tf.abs(z) <= 2
    return tf.math.abs(z)


# Function to draw Mandelbrot set for the entire complex plane
def draw_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = tf.linspace(xmin, xmax, width)
    r2 = tf.linspace(ymax, ymin, height)  # flipped
    c = tf.complex(r1[None, :], r2[:, None])  # swapped
    return mandelbrot_tf(c, max_iter)


# Function to convert from pixel to complex coordinates
def pixel_to_complex(x, y, width, height, xmin, xmax, ymin, ymax):
    real = xmin + (x / width) * (xmax - xmin)
    imag = ymax - (y / height) * (ymax - ymin)  # flipped
    return (real, imag)


class MandelbrotApp:
    def __init__(self, root):
        self.xmin, self.xmax, self.ymin, self.ymax = -2.0, 1.0, -1.5, 1.5
        self.width, self.height = 800, 800
        self.max_iter = 512
        self.coords = []
        self.rect = None

        self.canvas = tk.Canvas(root, width=self.width, height=self.height)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.onclick)
        self.canvas.bind("<B1-Motion>", self.onmotion)
        self.canvas.bind("<ButtonRelease-1>", self.onrelease)

        reset_button = tk.Button(root, text="Reset", command=self.reset)
        reset_button.pack()

        save_button = tk.Button(root, text="Save", command=self.save)
        save_button.pack()

    def redraw(self):
        z = draw_mandelbrot(self.xmin, self.xmax, self.ymin, self.ymax, self.width, self.height, self.max_iter)
        img_color = Image.fromarray(tf.cast(plt.cm.brg(z / tf.reduce_max(z)) * 255, tf.uint8).numpy())
        img_tk = ImageTk.PhotoImage(img_color)
        self.canvas.delete("all")  # Clear the canvas
        self.canvas.create_image(0, 0, anchor='nw', image=img_tk)
        self.canvas.image = img_tk
        return img_color

    def onclick(self, event):
        self.coords.append((event.x, event.y))
        self.update_rectangle(event.x, event.y)

    def onmotion(self, event):
        if self.coords:
            self.update_rectangle(event.x, event.y)

    def onrelease(self, event):
        self.coords.append((event.x, event.y))
        self.coords = [pixel_to_complex(*coord, self.width, self.height, self.xmin, self.xmax, self.ymin, self.ymax) for
                       coord in self.coords]
        self.xmin, self.xmax = sorted(coord[0] for coord in self.coords)
        self.ymin, self.ymax = sorted(coord[1] for coord in self.coords)  # sorted, not min and max
        self.coords.clear()
        self.redraw()

    def update_rectangle(self, x, y):
        if self.rect is not None:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(*self.coords[0], x, y, outline='red')

    def reset(self):
        self.xmin, self.xmax, self.ymin, self.ymax = -2.0, 1.0, -1.5, 1.5
        self.redraw()

    def save(self):
        img_color = self.redraw()
        img_color.save('mandelbrot_color.png')


def main():
    root = tk.Tk()
    app = MandelbrotApp(root)
    app.redraw()
    tk.mainloop()


if __name__ == '__main__':
    main()
