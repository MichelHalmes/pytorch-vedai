from PIL import Image

from os import path, listdir

import sys

from src import config

# Converts DOTA .png images to .jpg

def main():
    images_dir = path.dirname(path.join(config.DATA_PATH.format(name="dota"), config.IMAGES_PATH))
    image_ids = []
    for filename in sorted(listdir(images_dir)):
        if not filename.endswith(".png"):
            continue
        image_path = path.join(images_dir, filename)
        new_path = image_path[:-4]+".jpg"
        if path.isfile(new_path):
            continue
        print(image_path)
        image = Image.open(image_path)
        rgb_im = image.convert("RGB")
        rgb_im.save(new_path, quality=99)

if __name__ == "__main__":
    main()
