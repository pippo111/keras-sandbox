import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_confusions(image, mask, pred, coords, show=True, save=False, filename='confusion_plot.png'):
    axis0, axis1, axis2 = coords

    image_0 = np.rot90(image[axis0, :, :])
    image_1 = np.rot90(image[:, axis1, :])
    image_2 = np.rot90(image[:, :, axis2])
    
    mask_0 = np.rot90(mask[axis0, :, :])
    mask_1 = np.rot90(mask[:, axis1, :])
    mask_2 = np.rot90(mask[:, :, axis2])
    
    pred_0 = np.rot90(pred[axis0, :, :])
    pred_1 = np.rot90(pred[:, axis1, :])
    pred_2 = np.rot90(pred[:, :, axis2])

    combined_0 = mask_0 * 2 + pred_0
    combined_1 = mask_1 * 2 + pred_1
    combined_2 = mask_2 * 2 + pred_2
    
    fig, ax = plt.subplots(5, 3, figsize=(20, 20))
    
    ax[0][0].set_title('Scan images')
    ax[0][0].imshow(image_0, cmap='gray')
    ax[0][1].imshow(image_1, cmap='gray')
    ax[0][2].imshow(image_2, cmap='gray')
    
    ax[1][0].set_title('Original masks')
    ax[1][0].imshow(mask_0, cmap='gray')
    ax[1][1].imshow(mask_1, cmap='gray')
    ax[1][2].imshow(mask_2, cmap='gray')
    
    ax[2][0].set_title('Predicted masks')
    ax[2][0].imshow(pred_0, cmap='gray')
    ax[2][1].imshow(pred_1, cmap='gray')
    ax[2][2].imshow(pred_2, cmap='gray')

    cmap = matplotlib.colors.ListedColormap(['black', 'red', 'yellow', 'green'])
    norm = matplotlib.colors.Normalize(vmin=0, vmax=3)

    ax[3][0].set_title('Combined masks')
    ax[3][0].imshow(combined_0, cmap=cmap, norm=norm)
    ax[3][1].imshow(combined_1, cmap=cmap, norm=norm)
    ax[3][2].imshow(combined_2, cmap=cmap, norm=norm)

    ax[4][0].set_title('Overlayed scan')
    ax[4][0].imshow(image_0, cmap='gray')
    ax[4][0].imshow(combined_0, cmap=cmap, norm=norm, alpha=0.2)
    ax[4][1].imshow(image_1, cmap='gray')
    ax[4][1].imshow(combined_1, cmap=cmap, norm=norm, alpha=0.2)
    ax[4][2].imshow(image_2, cmap='gray')
    ax[4][2].imshow(combined_2, cmap=cmap, norm=norm, alpha=0.2)
    
    if show:
        plt.show()

    if save:
        fig.savefig(filename)

    plt.close('close')