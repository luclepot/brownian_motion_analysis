import argparse
import sys

parser = argparse.ArgumentParser(description='parser for pims_tracker')

parser.add_argument('--input', metavar='input_path', type=str, nargs=1, required=True, help='glob-string input for data')
parser.add_argument('--output', metavar='output_path', type=str, required=False, nargs=1, help='out path for saved data', default='NONE')

args = parser.parse_args()

import cv2
import numpy as np
import pandas as pd
import glob
import sys
import os
import pims
import trackpy as tp
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_PATH = os.path.dirname(os.path.realpath(__file__))
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

def loop_images(images, save_path):
    
    i,N = 0,len(images)

    sep_size = 20
    dpi = 100.
    rel_height = 2.

    figsize=(float(3*200 + 4*sep_size)/dpi, 200*rel_height/dpi)

    sep_val = np.ones_like(images[0])[:,:20]*255

    cv2.namedWindow('processed_image', cv2.WINDOW_KEEPRATIO)
    cv2.createTrackbar('Diameter', 'processed_image', 5, 20, lambda x: 0)
    cv2.createTrackbar('Minmass', 'processed_image', 500, 3000, lambda x: 0)
    cv2.createTrackbar('Separation', 'processed_image', 2, 40, lambda x: 0)
    cv2.createTrackbar('# Frames to Save', 'processed_image', 100, 1000, lambda x: 0)
    cv2.createTrackbar('lost particle memory', 'processed_image', 5, 30, lambda x: 0)

    cv2.resizeWindow('processed_image', 800,600)

    fig,ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect('equal')
    PAUSE = False
    SAVE_F_HISTORY = False
    f_history = None
    f_history_i = -1
    f_history_i0 = 0
    link = None

    ax.set_ylim(0,200)
    ax.set_xlim(0,200)
    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8,)
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plot = cv2.cvtColor(plot,cv2.COLOR_RGB2BGR)
            
    print('ENTERING CV2 INTERFACE')
    print('keybindings to know:')
    print(' - s: start save')
    print(' - spacebar: pause loop')
    print(' - esc: exit program')

    while True:
        if not PAUSE:
            proc = pims.as_gray(images[i])
            found = tp.locate(
                proc, 
                max([2*cv2.getTrackbarPos('Diameter', 'processed_image') + 1, 3]), 
                minmass=cv2.getTrackbarPos('Minmass', 'processed_image'),
                separation=cv2.getTrackbarPos('Separation', 'processed_image'),
            )

            # add located centers
            anot = images[i].copy()
            for center in zip(found.x.astype(int), found.y.astype(int)):
                cv2.circle(img=anot, center=center, radius=6, color=(0,255,0))

            if SAVE_F_HISTORY:
                t,coords = next(tp.linking.utils.coords_from_df(found, ['y', 'x'], 'frame'))
                if link is None:
                    f_history = pd.DataFrame()
                    f_history_i = 0
                    ax.clear()
                    link = tp.linking.Linker(
                        search_range=10,
                        memory=cv2.getTrackbarPos('lost particle memory', 'processed_image')
                    )
                    link.init_level(coords, f_history_i)
                else:
                    link.next_level(coords, f_history_i)
                found['particle'] = link.particle_ids
                f_history = pd.concat([f_history, found])
                f_history_i += 1
            if f_history is not None:
                ax.clear()
                tp.plot_traj(f_history, ax=ax)
                ax.set_ylim(200, 0)
                ax.set_xlim(0,200)

            if SAVE_F_HISTORY:
                fig.canvas.draw()
                plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8,)
                plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plot = cv2.cvtColor(plot,cv2.COLOR_RGB2BGR)
            
            to_show = np.concatenate([
                sep_val,
                images[i], 
                sep_val,
                np.tile(proc[:,:,None], (3)).astype(np.uint8),
                sep_val,
                anot.astype(np.uint8),
                # np.tile(anot[:,:,None], (3)),
                sep_val,
            ], axis=1)

            len_diff = to_show.shape[1] - plot.shape[1]
            if len_diff > 0:
                plot = np.concatenate([plot, np.ones_like(plot[:,:len_diff,:])*255], axis=1)
            elif len_diff < 0:
                to_show = np.concatenate([to_show, np.ones_like(to_show[:,:-len_diff,:])*255], axis=1)
            to_show = np.concatenate([to_show, plot])

            cv2.putText(
                to_show, 'frame {:03}/{:03}'.format(i, len(frames)), 
                org=(10,582), fontFace=cv2.FONT_HERSHEY_PLAIN, 
                fontScale=1.3, color=(0,0,0), thickness=2
            )

            cv2.imshow(
                'processed_image', 
                to_show,
            )
            # else:
            #     cv2.imshow(
            #         'processed_image',
            #         to_show
            #     )
            i += 1
            if i == N:
                i = 0
                ax.clear()
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord(' '):
            PAUSE = not PAUSE
        if k == ord('s') or f_history_i >= cv2.getTrackbarPos('# Frames to Save', 'processed_image') or f_history_i  == len(images):
            if SAVE_F_HISTORY:
                print('stopping save at {} frames'.format(f_history_i))
                
                if save_path == 'NONE':
                    now = datetime.datetime.now()
                    folder_name = '{}/data_out/{}'.format(_THIS_PATH, now.strftime('%Y_%b_%d'))
                    filename = '{}_{}-frames.csv'.format(now.strftime('%I:%M:%S-%p'), f_history.frame.nunique())

                    path = os.path.join(folder_name, filename)
                else:
                    if not save_path.endswith('.csv'):
                        save_path += '.csv'
                    path = save_path
                    folder_name = os.path.dirname(path)

                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)       

                print('Saving data to "{}"'.format(os.path.abspath(path)))
                print(' - {} frames'.format(f_history.frame.nunique()))
                print(' - {} particles'.format(f_history.particle.nunique()))
                print(' - {} MB'.format(f_history.memory_usage().sum()*1e-6))
                f_history.to_csv(path)                

                link = None
                f_history_i = 0
                f_history = None
            else:
                print('Beginning save!')
                link = None
                f_history_i = 0
                i = 0
                f_history = None

            SAVE_F_HISTORY = not SAVE_F_HISTORY
    
    cv2.destroyWindow('processed_image')
    return 0

if __name__ == '__main__':
    # images = load_images(load_info('brownian_motion/debug_data'))
    # print(args.input, args.output)
    load_path = os.path.abspath(args.input[0])
    save_path = os.path.abspath(args.output[0])
    print('loading images from path "{}"'.format(load_path))
    frames = pims.open(load_path)
    
    loop_images(frames, save_path)

    del frames
    cv2.destroyAllWindows()
    print('closed window')
