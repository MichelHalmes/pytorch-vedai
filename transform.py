
import numpy as np
from torchvision import transforms

IMG_SIZE = (300, 300)
RGB_AVG = (0.5, 0.5, 0.5)
RGB_STDDEV = (0.5, 0.5, 0.5)

def get_transform_fn(for_training):
    img_transforms = []

    if for_training:
        img_transforms.append(transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5))


    img_transforms.extend([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(RGB_AVG, RGB_STDDEV) # TODO
    ])

    img_transform = transforms.Compose(img_transforms)

    def transform(image, boxes, labels):
        # TODO: check it doesnt modify the original
        return img_transform(image), boxes, labels
    
    return transform


def inverse_img_transform(img_tensor):
    image = img_tensor.numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array(RGB_STDDEV) + np.array(RGB_AVG)
    return image



