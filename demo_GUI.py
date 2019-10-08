import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import torch
import torchvision
import numpy as np
import argparse
import threading
import queue

from models.M2TCC import CrowdCounter
from config import cfg
from misc import pytorch_ssim

#%%
class VideoCapture:
    def __init__(self, video_source=0):
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
    
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

class CC:
    def __init__(self, video_source):
        # setup the net
        torch.device("cuda")
        torch.cuda.set_device(0)
        torch.backends.cudnn.enabled = True
        self.net = CrowdCounter(cfg.GPU_ID,cfg.NET, torch.nn.MSELoss(),pytorch_ssim.SSIM(window_size=11))
        self.net.load_state_dict(torch.load(args.model))
        self.net.cuda()
        self.net.eval()
        self.mean, self.std = [0.53144014, 0.50784626, 0.47360169], [0.19302233, 0.18909324, 0.17572044]
        self.vid = VideoCapture(video_source)
        
    def run(self, thread_queue):
        global go_thread
        while go_thread:
            ret, frame = self.vid.get_frame()
            if ret:
                # convert to pytorch tensor, normalize and forward propagation
                tensor = torchvision.transforms.ToTensor()(frame)
                tensor = torchvision.transforms.functional.normalize(tensor, mean=self.mean, std=self.std)
                with torch.no_grad():
                       tensor = torch.autograd.Variable(tensor[None,:,:,:]).cuda()
                       pred_map = self.net.test_forward(tensor)
                pred_map = pred_map.cpu().data.numpy()[0,0,:,:]
                gray = np.repeat(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis], 3, axis=2)
                thread_queue.put((gray,pred_map))
                
        
class CCGUI:
    def __init__(self, window, window_title, set_list):
        self.window = window
        self.window.title(window_title)
        self.cmap = args.cmap
        
        # cc net class in another thread
        self.CCNet = CC(args.video)
        self.thread_queue = queue.Queue()
        self.CC_thread = threading.Thread(target=self.CCNet.run, kwargs={'thread_queue':self.thread_queue})
        self.CC_thread.start()

        # canvas for video and widgets
        self.canvas = tk.Canvas(window, width = self.CCNet.vid.width, height = self.CCNet.vid.height)
        self.canvas.pack()

        # menu to choose blend mode
        self.blendlabel = tk.Label(self.window)
        self.blendlabel.config(text = "Blend mode:")
        self.blendlabel.pack(side=tk.LEFT, expand=True, anchor=tk.E)
        self.blendmodes = ['Add','Lighten','Mix','Multiply']
        self.blend_var = tk.StringVar()
        self.blend_var.set(args.mode)
        self.blend_menu = tk.OptionMenu(self.window, self.blend_var, *self.blendmodes)
        self.blend_menu.pack(side=tk.LEFT, expand=False, anchor=tk.CENTER)
        
        # menu to choose colormap
        self.cmaplabel = tk.Label(self.window)
        self.cmaplabel.config(text = "      Colormap:")
        self.cmaplabel.pack(side=tk.LEFT, expand=False, anchor=tk.E)
        self.cmaplist = ['Autumn','Bone','Jet','Winter','Rainbow','Ocean','Summer','Spring','Cool','HSV','Pink','Hot']
        self.cmap_var = tk.StringVar()
        self.cmap_var.set(args.cmap)
        self.cmap_menu = tk.OptionMenu(self.window, self.cmap_var, *self.cmaplist)
        self.cmap_menu.pack(side=tk.LEFT, expand=True, anchor=tk.W)
        
        # text to display crowd count
        self.cclabel = tk.Label(self.window)
        self.cclabel.config(text = "TOT count: ", font = ('calibri',(30)))
        self.cclabel.pack(side=tk.LEFT, expand=True)
        
        # slider to select opacity
        self.opac_var = tk.DoubleVar()
        self.opac_var.set(args.opacity*100)
        self.opac_slider = tk.Scale(self.window, variable=self.opac_var, orient = tk.HORIZONTAL, label='Opacity of density map:', length = 200)
        self.opac_slider.pack(side=tk.LEFT, expand=True)
        
        
        # updates window every delay ms
        self.delay = 100
        self.update()
        self.window.mainloop()

    def update(self):
        try:
            # gets prediction and original frame from the crowd counter thread, then overlaps according to blend mode
            gray, pred_map = self.thread_queue.get(0)
            cmap = self.cmap_var.get()
            dmap = cv2.cvtColor(cv2.applyColorMap(np.array(pred_map/np.max(pred_map+1e-20) * 255, dtype = np.uint8), self.cmaplist.index(cmap)), cv2.COLOR_BGR2RGB)
            blend = self.blend_var.get()
            opacity = self.opac_var.get()/100
            dmap = dmap.astype(np.float32)/255
            gray = gray.astype(np.float32)/255
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
            # write total people count below image and show results
            self.cclabel.config(text = "TOT count: "+str((int(np.sum(pred_map)/100 +0.5))))
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.res))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            self.window.after(self.delay, self.update)
        except queue.Empty:
            self.window.after(self.delay, self.update)
      

if __name__=='__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./checkpoints/venice_all_ep_1286_mae_5.9_mse_7.4.pth", type=str, help="Weights file to use.")
    parser.add_argument("--video", default="D:/Alex/venice/test_data/videos/4895.mp4", type=str, help="Video file to elapse or camera ip.")
    parser.add_argument("--opacity", default=0.7, type=float, help="Opacity value for the density map overlap.")
    parser.add_argument("--cmap", default='Hot', type=str, help="Colormap to use.", choices = ['Autumn','Bone','Jet','Winter','Rainbow','Ocean','Summer','Spring','Cool','HSV','Pink','Hot'])
    parser.add_argument("--mode", default='Add', type=str, help="Blend mode to use.", choices = ['Add','Lighten','Mix','Multiply'])
    args = parser.parse_args()
    
    go_thread = True
    CCGUI(tk.Tk(), "Crowd Counting", args)
    go_thread = False
    
    