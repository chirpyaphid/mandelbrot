# mandelbrot
Two implementations of generating mandelbrot images using python.

There are two versions here mb just uses python and multiproccessing to improve the calculation times when it comes to generating the mandelbrot but is CPU bound.

mb_tensoflow_version uses tensorflow to do the heavy lifting when it comes to the calculations by using a GPU if present and if it is supported.

tkinter is used to display the image and provide a UI for allowing the zooming into the mandelbrot by clicking and dragging over the area you want to zoom into.
