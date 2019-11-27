import matplotlib.pyplot as plt
import matplotlib.patches as patches


from transform import inverse_img_transform

def plot_sample(img_tensor, boxes, lables):
    image = inverse_img_transform(img_tensor)
    plt.imshow(image)
    ax = plt.gca()
    img_width, img_height, _ = image.shape
    for box, label in zip(boxes, lables):
        cx, cy, w, h = box
        x_min = cx*img_width - w*img_width/2
        y_min = cy*img_height - h*img_height/2
        rect = patches.Rectangle((x_min, y_min), w*img_width, h*img_height, 
                                linewidth=1, edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    plt.show()