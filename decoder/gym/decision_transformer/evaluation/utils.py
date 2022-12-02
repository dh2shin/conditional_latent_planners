import os
import imageio
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(im) < 128:
        text_color = (255,255,255)
    else:
        text_color = (0,0,0)
    drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {episode_num+1}', fill=text_color)

    return im

def _maze_evaluation_pts(dataset, brownian):
    model_type = 'brownian' if brownian else 'base'
    return f'decoder/gym/weights/maze/{dataset}/{model_type}.pt'

def _adroit_evaluation_pts(expert, dataset, brownian):
    dataset_type = 'expert' if expert == "expert" else 'human'
    model_type = 'brownian' if brownian else 'base'
    return f'decoder/gym/weights/adroit/{dataset}/{dataset_type}-{model_type}.pt'

def _maze_mlp_pts(dataset, brownian):
    model_type = 'brownian' if brownian else 'base'
    return f'decoder/gym/weights/maze/{dataset}/mlp_state.pt'

def _adroit_mlp_pts(expert, dataset, brownian):
    dataset_type = 'expert' if expert == "expert" else 'human'
    model_type = 'brownian' if brownian else 'base'
    return f'decoder/gym/weights/adroit/{dataset}/{dataset_type}_mlp_state.pt'
