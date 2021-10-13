import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys
import os
import argparse

## helper functions

def load_info(data_path):
    # loads dataframe with all image info (frame #/path) in a given directory. can add more stuff later
    info = pd.DataFrame(glob.glob('{}/*.bmp'.format(data_path)), columns=['path'])
    info['frame'] = info.path.str.replace('.bmp', '').str.split('Frame ').str[1].astype(int)
    info.set_index('frame', inplace=True)
    return info.sort_index()

def load_images(info_df, max_frames=None, max_bytes=None,):    
    if not len(info_df) > 0:
        raise Exception('Info dataframe empty')

    test_img = cv2.imread(info_df.path.iloc[0])
    shape = test_img.shape
    size_bytes = sys.getsizeof(test_img)
    N = len(info_df)
    if max_bytes is not None:
        N = min([N, int(np.floor(max_bytes/size_bytes))])
    if max_frames is not None:
        N = min([N, max_frames])
    
    images = np.empty(shape=(N,)+shape,dtype=test_img.dtype)
    for i in range(N):
        images[i] = cv2.imread(info_df.path.iloc[i])
    
    del test_img 
    return images

class image_analyzer:
    def __init__(self):
        self._params = {}
        pass

    def process(self, img):
        return img

    def update_params(self, k):
        pass

    def get_params(self):
        return self._params

    

def loop_images(images):
    
    i,N = 0,len(images)

    sep_size = 20
    dpi = 100.
    rel_height = 1.

    figsize=(float(3*200 + 4*sep_size)/dpi, 200*rel_height/dpi)

    sep_val = np.ones_like(images[0])[:,:20]*255

    cv2.namedWindow('processed_image', cv2.WINDOW_KEEPRATIO)
    cv2.createTrackbar('Threshold', 'processed_image', 100, 255, lambda x: 0)
    cv2.resizeWindow('processed_image', 800,600)

    fig,ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    PAUSE = False
    
    while cv2.getWindowProperty('processed_image', cv2.WND_PROP_VISIBLE) > 0:
        if not PAUSE:
            gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            proc = cv2.threshold(
                    gray,
                    cv2.getTrackbarPos('Threshold', 'processed_image'),
                    255,  cv2.THRESH_BINARY
                )[1]

            ax.clear()
            ax.hist(np.random.normal(size=10000), bins=100)
            ax.set_ylim(-2,500)
            ax.set_xlim(-3,3)
            ax.set_ylabel('Count')
            plt.yticks(fontsize=5)
            fig.canvas.draw()

            plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,)
            plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plot = cv2.cvtColor(plot,cv2.COLOR_RGB2BGR)        

            to_show = np.concatenate([
                sep_val,
                images[i], 
                sep_val,
                np.tile(gray[:,:,None], (3)),
                sep_val,
                np.tile(proc[:,:,None], (3)),
                sep_val,
            ], axis=1)

            cv2.imshow(
                'processed_image', 
                # to_show,
                np.concatenate([to_show, plot])
            )
            i += 1
            if i == N:
                i = 0
        
        k = cv2.waitKey(50) & 0xFF
        if k == 27:
            break
        if k == ord(' '):
            PAUSE = not PAUSE
    
    cv2.destroyWindow('processed_image')
    return 0

if __name__ == '__main__':
    images = load_images(load_info('brownian_motion/debug_data'))
    loop_images(images)