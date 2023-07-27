import numpy as np
from numba import jit
from PIL import Image, ImageTk
import tkinter as tk
from multiprocessing import Pool
import matplotlib.pyplot as plt

# function to generate Mandelbrot set
@jit(nopython=True)
def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

# function to draw Mandelbrot set for a portion of the complex plane
def draw_mandelbrot_part(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    return np.array([[mandelbrot(complex(r, i),max_iter) for r in r1] for i in r2])

# function to draw Mandelbrot set for the entire complex plane
def draw_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, pool):
    xstep = (xmax - xmin) / 5
    ystep = (ymax - ymin) / 5
    results = pool.starmap(draw_mandelbrot_part, [(xmin+xstep*i, xmin+xstep*(i+1), ymin+ystep*j, ymin+ystep*(j+1), width//5, height//5, max_iter) for j in range(5) for i in range(5)])
    return np.block([[results[i+j*5] for i in range(5)] for j in range(5)])

# function to convert from pixel to complex coordinates
def pixel_to_complex(x, y, width, height, xmin, xmax, ymin, ymax):
    real = xmin + (x / width) * (xmax - xmin)
    imag = ymin + (y / height) * (ymax - ymin)
    return (real, imag)

def main():
    # initial parameters
    xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
    width, height = 800, 800
    max_iter = 256

    # tkinter window
    root = tk.Tk()

    # tkinter canvas
    canvas = tk.Canvas(root, width=width, height=height)
    canvas.pack()

    # event handlers
    coords = []
    rect = None
    def onclick(event):
        nonlocal rect
        coords.append((event.x, event.y))
        if rect is not None:
            canvas.delete(rect)
        rect = canvas.create_rectangle(*coords[0], *coords[0], outline='red')

    def onmotion(event):
        nonlocal rect
        if coords:
            if rect is not None:
                canvas.delete(rect)
            rect = canvas.create_rectangle(*coords[0], event.x, event.y, outline='red')

    def onrelease(event):
        nonlocal xmin, xmax, ymin, ymax
        coords.append((event.x, event.y))
        coords[0] = pixel_to_complex(*coords[0], width, height, xmin, xmax, ymin, ymax)
        coords[1] = pixel_to_complex(*coords[1], width, height, xmin, xmax, ymin, ymax)
        xmin, xmax = min(coords[0][0], coords[1][0]), max(coords[0][0], coords[1][0])
        ymin, ymax = min(coords[0][1], coords[1][1]), max(coords[0][1], coords[1][1])
        coords.clear()
        redraw()

    def redraw():
        with Pool() as pool:
            z = draw_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter, pool)
        img_color = Image.fromarray(np.uint8(plt.cm.viridis(z/np.max(z)) * 255))
        img_gray = Image.fromarray(np.uint8(z * 255), 'L')
        img_tk = ImageTk.PhotoImage(img_color)
        canvas.create_image(0, 0, anchor='nw', image=img_tk)
        canvas.image = img_tk
        return img_color, img_gray

    # connect event handlers
    canvas.bind("<Button-1>", onclick)
    canvas.bind("<B1-Motion>", onmotion)
    canvas.bind("<ButtonRelease-1>", onrelease)

    # reset button
    def reset():
        nonlocal xmin, xmax, ymin, ymax
        xmin, xmax, ymin, ymax = -2.0, 1.0, -1.5, 1.5
        redraw()

    reset_button = tk.Button(root, text="Reset", command=reset)
    reset_button.pack()

    # save button
    def save():
        img_color, img_gray = redraw()
        img_color.save('mandelbrot_color.png')
        img_gray.save('mandelbrot_gray.png')

    save_button = tk.Button(root, text="Save", command=save)
    save_button.pack()

    # initial draw
    redraw()

    # start event loop
    tk.mainloop()

if __name__ == '__main__':
    main()
