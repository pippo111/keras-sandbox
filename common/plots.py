import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 45, 50, 72
# 25, 20, 25

# 25, 23, 40

def save_sample_plot(image, mask, pred, filename):
    image_0 = np.rot90(image[25, :, :])
    image_1 = np.rot90(image[:, 23, :])
    image_2 = np.rot90(image[:, :, 40])
    
    mask_0 = np.rot90(mask[25, :, :])
    mask_1 = np.rot90(mask[:, 23, :])
    mask_2 = np.rot90(mask[:, :, 40])
    
    pred_0 = np.rot90(pred[25, :, :])
    pred_1 = np.rot90(pred[:, 23, :])
    pred_2 = np.rot90(pred[:, :, 40])

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
    
    fig.savefig(filename)
    plt.close()