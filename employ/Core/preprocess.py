import cv2
import numpy as np
from PIL import Image, ImageEnhance


# def preprocess(image, contrast_value):
#     image = adjust_contrast(image, contrast_value)
#     image
#
#
# def adjust_contrast(image, contrast_value):
#
#     if contrast_value > 50:
#         contrast_value = contrast_value - 50
#         lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#         clahe = cv2.createCLAHE(clipLimit=contrast_value * 0.05, tileGridSize=(5, 5))
#         lab[:, :, 0] = clahe.apply(lab[:, :, 0])
#         image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#     elif contrast_value < 50:
#         alpha = (contrast_value + 50) / 100
#         beta = np.max(image) * (1 - (contrast_value + 50) / 100)
#         # beta = 0
#         image = image * alpha + beta
#     # import pdb; pdb.set_trace()
#     return image
# #
# def adjust_contrast(image, contrast_value):
#     contrast_value = 100 - contrast_value
#     image= image.astype(np.float)
#     if contrast_value >50:
#         coefficient = np.array([-0.5, -0.5, -2.0])
#         image = image + (contrast_value - 50) * coefficient
#     elif contrast_value < 50:
#         coefficient = np.array([-0.1, -0.1, -0.4])
#         image = image + (contrast_value-50)*coefficient
#     return image


# def adjust_contrast(image, contrast_value):
#     if contrast_value == 50:
#         return image
#     elif contrast_value>50:
#         rgb = [-0.5, -0.5, -2.0]
#     else:
#         rgb = [-0.1, -0.1, -0.4]
#     for i in range(image.shape[0]):
#         image[i] = image[i] + ((50-contrast_value)*rgb[i]*(2. / 255))
#     return image

# def adjust_contrast(image, contrast_value):
#     if contrast_value == 50:
#         return image
#     elif contrast_value<50:
#         rgb = np.array([-0.5, -0.5, -2.0])
#     else:
#         rgb = np.array([-0.1, -0.1, -0.4])
#     image = image + (50-contrast_value)*rgb
#     return image
#
# def clahe(image, clip_limit=0.3):
#     lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#     clahe = cv2.createCLAHE(clipLimit=clip_limit * 0.05, tileGridSize=(5, 5))
#     lab[:, :, 0] = clahe.apply(lab[:, :, 0])
#     image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#     return image


def adjust_contrast(image , contrast_val=50):

    adjust_val = 1+(1.5*(contrast_val-50)/100)

    im = Image.fromarray(image.astype('uint8')).convert('RGB')
    img = Image.fromarray(image.astype('uint8')).convert('HSV')

    Saturation = np.array(img)[:, :, 1]
    mean_Saturation = np.mean(Saturation)

    if mean_Saturation < 17:

        enhancer = ImageEnhance.Contrast(im)
        factor = 2 * adjust_val # increase contrast
        im_output = enhancer.enhance(factor)

        image = np.array(im_output)
    else:
        image = np.array(im)
    return image


