from PIL import Image

from os import path, listdir

import config

images_dir = path.dirname(path.join(config.DATA_PATH.format(name="dota"), config.IMAGES_PATH))
image_ids = []
for filename in sorted(listdir(images_dir)):
    if not filename.endswith(".png"):
        continue
    image_path = path.join(images_dir, filename)
    print(image_path)
    image = Image.open(image_path)
    rgb_im = image.convert('RGB')
    new_path = image_path[:-4]+".jpg"
    rgb_im.save(new_path, quality=99)