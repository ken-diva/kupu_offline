import json, io, logging
import uuid
import numpy as np

from PIL import Image

logger = logging.getLogger(__name__)

CLOUD_STORAGE_BUCKET = 'kupu-364702.appspot.com'

### Brush Export ###

class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i:self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """ from bytes array to bits by num position
    """
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """ get bit string from bytes data
    """
    return ''.join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def decode_rle(rle, print_params: bool = False):
    """ from LS RLE to numpy uint8 3d image [width, height, channel]
    
    Args:
        print_params (bool, optional): If true, a RLE parameters print statement is suppressed
    """
    input = InputStream(bytes2bit(rle))
    num = input.read(32)
    word_size = input.read(5) + 1
    rle_sizes = [input.read(4) + 1 for _ in range(4)]
    
    if print_params:
        print('RLE params:', num, 'values', word_size, 'word_size', rle_sizes, 'rle_sizes')
        
    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = input.read(1)
        j = i + 1 + input.read(rle_sizes[input.read(2)])
        if x:
            val = input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = input.read(word_size)
                out[i] = val
                i += 1
    return out

# create gambar disini
def convert_to_png(path, pos):
  """ Convert the .json from brushlabels
  
  Args: 
    json_path = path to the json file from frontend module
    pos = ANT or POST (depan / belakang)
  """
  f = open(path)
  data = json.load(f)

  new_image = []
  for d in data:

    width = d['original_width']
    height = d['original_height']
    brushlabels = str(d['value']['brushlabels'])

    img_name = brushlabels.translate({ord(i): None for i in "[' ']"})
    decoded_rle = np.reshape(decode_rle(d['value']['rle']), [height, width, 4])[:, :, 3]

    img = Image.fromarray(decoded_rle).convert('RGBA')

    img_data = img.getdata()

    # for coloring purpose
    if img_name == 'Skull':
        r, g, b, a = 21, 193, 78, 125
    elif img_name == 'CervicalVert':
        r, g, b, a = 0, 121, 255, 125
    elif img_name == 'ThoracicVert':
        r, g, b, a = 0, 228, 255, 125
    elif img_name == 'Clavicle':
        r, g, b, a = 109, 21, 193, 125
    elif img_name == 'Ribs':
        r, g, b, a = 228, 70, 206, 125
    elif img_name == 'Scapula':
        r, g, b, a = 224, 131, 46, 125
    elif img_name == 'Sternum':
        r, g, b, a = 129, 64, 4, 125
    elif img_name == 'Humerus':
        r, g, b, a = 14, 24, 156, 125
    elif img_name == 'LumbarVert':
        r, g, b, a = 166, 5, 29, 125
    elif img_name == 'Sacrum':
        r, g, b, a = 255, 122, 0, 125
    elif img_name == 'Pelvis':
        r, g, b, a = 16, 119, 7, 125
    elif img_name == 'Femur':
        r, g, b, a = 225, 235, 52, 125
    else:
        r, g, b, a = 255, 255, 255, 125
    
    # # only if using python 3.10
    # match img_name:
    #     case "Skull":
    #       r, g, b, a = 21, 193, 78, 125
    #     case "CervicalVert":
    #       r, g, b, a = 0, 121, 255, 125
    #     case "ThoracicVert":
    #       r, g, b, a = 0, 228, 255, 125
    #     case "Clavicle":
    #       r, g, b, a = 109, 21, 193, 125
    #     case "Ribs":
    #       r, g, b, a = 228, 70, 206, 125
    #     case "Scapula":
    #       r, g, b, a = 224, 131, 46, 125
    #     case "Sternum":
    #       r, g, b, a = 129, 64, 4, 125
    #     case "Humerus":
    #       r, g, b, a = 14, 24, 156, 125
    #     case "LumbarVert":
    #       r, g, b, a = 166, 5, 29, 125
    #     case "Sacrum":
    #       r, g, b, a = 255, 122, 0, 125
    #     case "Pelvis":
    #       r, g, b, a = 16, 119, 7, 125
    #     case "Femur":
    #       r, g, b, a = 225, 235, 52, 125
    #     case _:
    #       r, g, b, a = 255, 255, 255, 125

    # ini logic nya ke else dulu
    if new_image:
        i = 0
        for item in img_data:
            if item[0] in list(range(85, 256)):
                new_image[i] = ((r,g,b,a))

            i = i + 1

    else:
        for item in img_data:
            if item[0] in list(range(10, 256)):
                new_image.append((r,g,b,a))
            else:
                new_image.append((255,255,255,0))

    # update image data
    img.putdata(new_image)

    # save new image
    if(pos == 'back'):
      img.save(f"./static/img_annotation/back/{img_name}.png")
      # img.save("image_back_done.png")
    else:
      img.save(f"./static/img_annotation/front/{img_name}.png")
      # img.save("image_front_done.png")

    new_image = []
