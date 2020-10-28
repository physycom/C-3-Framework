#This script runs SANet on a video source and shows result in a user friendly GUI.

#%% IMPORTS

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import torch
import torchvision
import numpy as np
import argparse
import threading
import queue
import platform

from models.M2TCC import CrowdCounter
from config import cfg
from misc import pytorch_ssim

OS = platform.system()

#%% DEFINITIONS

class VideoCapture:
    """ Class for capturing video from file or cam (threaded) """
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise RuntimeError("Unable to open video source", video_source)
        self.__w = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.__h = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #convert to rgb cause opencv uses bgr by default
        else:
            raise RuntimeError("Unable to capture frame / end of video")

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


class CC:
    """ Class for crowd counting with SANet (threaded) """
    def __init__(self, model):
        # Setup the net
        torch.device("cuda")
        torch.cuda.set_device(0)
        torch.backends.cudnn.enabled = True
        self.net = CrowdCounter(cfg.GPU_ID,cfg.NET, torch.nn.MSELoss(),pytorch_ssim.SSIM(window_size=11))
        self.net.load_state_dict(torch.load(model))
        self.net.cuda()
        self.net.eval()
        # IMPORTANT: when changing model, make sure you change config for the dataset it was trained in to make sure mean and std are correct
        self.mean, self.std = cfg.MEAN_STD
        # Print net structure
        # print(self.net)

    def run(self, stop_event, cam_queue, net_queue):
        while not stop_event.is_set():
            try:
                frame = cam_queue.get(0)
                # convert to pytorch tensor, normalize
                tensor = torchvision.transforms.ToTensor()(frame)
                tensor = torchvision.transforms.functional.normalize(tensor, mean=self.mean, std=self.std)
                # padding to multiples of 8
                xr = (8-tensor.shape[2]%8)%8
                yr = (8-tensor.shape[1]%8)%8
                tensor = torch.nn.functional.pad(tensor, (xr,xr,yr,yr), 'constant',0)
                # forward propagation
                with torch.no_grad():
                        tensor = torch.autograd.Variable(tensor[None,:,:,:]).cuda()
                        pred_map = self.net.test_forward(tensor)
                pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
                gray = np.repeat(np.array(frame.convert("L"))[:, :, np.newaxis], 3, axis=2)
                gray = gray.astype(np.float32)/255
                net_queue.put((gray, pred_map))
            except queue.Empty:
                pass

class AutoScrollbar(ttk.Scrollbar):
    """ A scrollbar that hides itself if it's not needed. Works only for grid geometry manager """
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
            ttk.Scrollbar.set(self, lo, hi)

class CanvasImage:
    """ Class for displaying a zoomable image """
    def __init__(self, placeholder, img):
        """ Initialize the ImageFrame """
        self.imscale = 1.0  # scale for the canvas image zoom, public for outer classes
        self.__delta = 1.3  # zoom magnitude
        self.__filter = Image.ANTIALIAS  # could be: NEAREST, BILINEAR, BICUBIC and ANTIALIAS
        # Create ImageFrame in placeholder widget
        self.__imframe = ttk.Frame(placeholder)  # placeholder of the ImageFrame object
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars. Public for outer classes
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0, xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        # Bind events to the Canvas
        self.canvas.bind('<Configure>', lambda event: self.__show_image())  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.__move_from)  # remember canvas position
        self.canvas.bind('<B1-Motion>',     self.__move_to)  # move canvas to the new position
        self.canvas.bind('<MouseWheel>', self.__wheel)  # zoom for Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>',   self.__wheel)  # zoom for Linux, wheel scroll down
        self.canvas.bind('<Button-4>',   self.__wheel)  # zoom for Linux, wheel scroll up

        self.__image = img  # load first image
        self.imwidth, self.imheight = self.__image.size  # public for outer classes
        self.tile_coords = (0,0, self.imwidth, self.imheight)
        self.__min_side = min(self.imwidth, self.imheight)  # get the smaller image side

        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.imwidth, self.imheight), width=0)
        self.__show_image()  # show image on the canvas
        self.canvas.focus_set()  # set focus on the canvas

    # Geometry management
    def grid(self, **kw):
        """ Put CanvasImage widget on the parent widget """
        self.__imframe.grid(**kw)  # place CanvasImage widget on the grid
        self.__imframe.grid(sticky='nswe')  # make frame container sticky
        self.__imframe.rowconfigure(0, weight=1)  # make canvas expandable
        self.__imframe.columnconfigure(0, weight=1)
    def pack(self, **kw):
        raise Exception('Cannot use pack with the widget ' + self.__class__.__name__)
    def place(self, **kw):
        raise Exception('Cannot use place with the widget ' + self.__class__.__name__)

    # Scrolling management
    def __scroll_x(self, *args, **kwargs):
        self.canvas.xview(*args)  # scroll horizontally
        self.__show_image()  # redraw the image
    def __scroll_y(self, *args, **kwargs):
        self.canvas.yview(*args)  # scroll vertically
        self.__show_image()  # redraw the image

    # Dragging management
    def __move_from(self, event):
        """ Remember previous coordinates for dragging with the mouse """
        self.canvas.scan_mark(event.x, event.y)
    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.__show_image()  # zoom tile and show it on the canvas
    def outside(self, x, y):
        """ Checks if the point (x,y) is outside the image area """
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            return False  # point (x,y) is inside the image area
        else:
            return True  # point (x,y) is outside the image area

    # Zooming management
    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        if self.outside(x, y): return  # zoom only inside image area
        scale = 1.0
        if OS == 'Darwin':
            if event.delta<0:  # scroll down, zoom out
                if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
                self.imscale /= self.__delta
                scale        /= self.__delta
            if event.delta>0:  # scroll up, zoom in, bigger
                i = float(min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1)
                if i < self.imscale: return  # 1 pixel is bigger than the visible area
                self.imscale *= self.__delta
                scale        *= self.__delta
        else:
            # Respond to Linux (event.num) or Windows (event.delta) wheel event
            if event.num == 5 or event.delta == -120:  # scroll down, zoom out
                if round(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
                self.imscale /= self.__delta
                scale        /= self.__delta
            if event.num == 4 or event.delta == 120:  # scroll up, zoom in, bigger
                i = float(min(self.canvas.winfo_width(), self.canvas.winfo_height()) >> 1)
                if i < self.imscale: return  # 1 pixel is bigger than the visible area
                self.imscale *= self.__delta
                scale        *= self.__delta
        self.canvas.scale('all', x, y, scale, scale)  # rescale all objects
        # Redraw some figures before showing image on the screen
        self.__show_image()

    def set_image(self,img):
        self.__image = img
        self.zoom_n_crop()

    def zoom_n_crop(self):
        """ Implements correct image zoom and cropping """
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0), self.canvas.canvasy(0), # get visible area of the canvas
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if  box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0]  = box_img_int[0]
            box_scroll[2]  = box_img_int[2]
        # Vertical part of the image is in the visible area
        if  box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1]  = box_img_int[1]
            box_scroll[3]  = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # zoom and crop in the visible area
            self.tile_coords = (int(x1 / self.imscale), int(y1 / self.imscale), int(x2 / self.imscale), int(y2 / self.imscale))
            image = self.__image.crop(self.tile_coords)
            self.image_tile = image.resize((int(x2 - x1), int(y2 - y1)), self.__filter)
            self.bbox_x = max(box_canvas[0], box_img_int[0])
            self.bbox_y = max(box_canvas[1], box_img_int[1])

    def __show_image(self):
        """ Show image on the Canvas """
        self.zoom_n_crop()
        imageid = self.canvas.create_image(self.bbox_x, self.bbox_y, anchor='nw', image=ImageTk.PhotoImage(image = self.image_tile))
        self.canvas.lower(imageid)  # set image into background


class MainWindow(ttk.Frame):
    """ Main window class that handles the GUI """
    def __init__(self, mainframe, args):
        """ Initialize the main Frame """
        ttk.Frame.__init__(self, master=mainframe)
        self.master.title('Crowd Counter')
        self.master.geometry('1300x800')  # size of the main window
        self.master.rowconfigure(0, weight=1)  # make the CanvasImage widget expandable
        self.master.columnconfigure(0, weight=1)

        self.vidc = VideoCapture(args.video)
        self.CCNet = CC(args.model)

        # Canvas with zoomable image
        ret, first_frame = self.vidc.vid.read()
        if not ret:
            raise RuntimeError("Error reading video")
        img = Image.fromarray(cv2.cvtColor(first_frame,cv2.COLOR_BGR2RGB))
        self.canvas = CanvasImage(self.master, img)
        self.canvas.grid(row=0,columnspan=6)

        # Background threads for video capture and network
        self.kill_thread = threading.Event()
        self.cam_queue = queue.Queue(maxsize=1)
        self.net_queue = queue.Queue()
        self.net_thread = threading.Thread(target=self.CCNet.run, args=(self.kill_thread, self.cam_queue, self.net_queue))
        self.net_thread.start()

        # Text to display crowd count
        self.cclabel = tk.Label(self.master)
        self.cclabel.config(text = "TOT count: ", font = ('calibri',(30)))
        self.cclabel.grid(row=1, column=0)
        # Slider to select opacity
        self.opac_var = tk.DoubleVar()
        self.opac_var.set(args.opacity*100)
        self.opac_slider = tk.Scale(self.master, variable=self.opac_var, orient = tk.HORIZONTAL, label='Opacity of density map:', length = 200)
        self.opac_slider.grid(row=1,column=1,padx=10)
        # Menu to choose blend mode
        self.blendlabel = tk.Label(self.master)
        self.blendlabel.config(text = "Blend mode:")
        self.blendlabel.grid(row=1,column=2,sticky='E')
        self.blendmodes = ['Add','Lighten','Mix','Multiply']
        self.blend_var = tk.StringVar()
        self.blend_var.set(args.mode)
        self.blend_menu = tk.OptionMenu(self.master, self.blend_var, *self.blendmodes)
        self.blend_menu.grid(row=1,column=3,sticky='W',padx=10)
        # Menu to choose colormap
        self.cmaplabel = tk.Label(self.master)
        self.cmaplabel.config(text = "Colormap:")
        self.cmaplabel.grid(row=1,column=4,sticky='E')
        self.cmaplist = ['Autumn','Bone','Jet','Winter','Rainbow','Ocean','Summer','Spring','Cool','HSV','Pink','Hot']
        self.cmap_var = tk.StringVar()
        self.cmap_var.set(args.cmap)
        self.cmap_menu = tk.OptionMenu(self.master, self.cmap_var, *self.cmaplist)
        self.cmap_menu.grid(row=1,column=5,sticky='W',padx=10)

        # Update window every delay ms
        self.delay = 20
        self.update()

    def update(self):
        try:
            # get prediction and gray frame from the crowd counter thread
            gray, pred_map = self.net_queue.get(0)
            pred_map = pred_map[:gray.shape[0],:gray.shape[1]]
            # create density map with the chosen colormap
            dmap = cv2.applyColorMap(np.array(pred_map/np.max(pred_map+1e-20) * 255, dtype = np.uint8), self.cmaplist.index(self.cmap_var.get()))
            dmap = (cv2.cvtColor(dmap, cv2.COLOR_BGR2RGB)).astype(np.float32)/255
            # overlap original image converted to grayscale with density map
            blend = self.blend_var.get()
            opacity = self.opac_var.get()/100
            if blend == 'Add':
                self.res = dmap*opacity + gray
            elif blend == 'Lighten':
                self.res = np.maximum(gray, dmap*opacity)
            elif blend == 'Mix':
                self.res = dmap*opacity + gray*(1-opacity)
            elif blend == 'Multiply':
                self.res = gray*(dmap*opacity+(1-opacity))
            else:
                raise Exception("No such blend mode found.")
            self.res = (np.clip(self.res,0,1)*255).astype(np.uint8)
            # write total people count below image and show result
            self.cclabel.config(text = "TOT count: " + str((int(np.sum(pred_map)/100 +0.5))))
            self.imageout= ImageTk.PhotoImage(image=Image.fromarray(self.res))
            self.canvas.canvas.create_image(self.canvas.bbox_x, self.canvas.bbox_y, anchor='nw', image = self.imageout)
            self.master.after(self.delay, self.update)
        except queue.Empty:
            # get new frame, crop it and send it to network
            self.canvas.set_image(self.vidc.get_frame())
            self.cam_queue.put(self.canvas.image_tile)
            self.master.after(self.delay, self.update)

#%% MAIN
if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./checkpoints/venice_all_ep_1286_mae_5.9_mse_7.4.pth", type=str, help="Weights file to use. Careful: if using different model you may need to change the trained dataset in config file.")
    parser.add_argument("--video", default="../venice/test_data/videos/4895.mp4", type=str, help="Video file to elapse or camera ip.")
    parser.add_argument("--opacity", default=0.7, type=float, help="Opacity value for the density map overlap.")
    parser.add_argument("--cmap", default='Hot', type=str, help="Colormap to use.", choices = ['Autumn','Bone','Jet','Winter','Rainbow','Ocean','Summer','Spring','Cool','HSV','Pink','Hot'])
    parser.add_argument("--mode", default='Add', type=str, help="Blend mode to use.", choices = ['Add','Lighten','Mix','Multiply'])
    args = parser.parse_args()

    app = MainWindow(tk.Tk(), args)
    app.mainloop()
    app.kill_thread.set()
    app.net_thread.join()