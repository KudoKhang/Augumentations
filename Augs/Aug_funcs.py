import cv2
import numpy as np
import albumentations as A
import os
import random

class Augment:
    def __init__(self):
        pass

    # Pixel-level transforms
    def ColorJitter(image, brightness=0.9, contrast=0.9, saturation=0.2, hue=0.2, p=1):
        """
        :param image:
        :param brightness:
        :param contrast:
        :param saturation:
        :param hue:
        :param p: xác suất thực hiện phép augment
        :return:
        """
        aug = A.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.2, hue=0.2, p=1)
        return aug(image=image)['image']

    def AdvanceBlur(image, blur_limit=(3, 7), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0), rotate_limit=90, beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1), p=1):
        """
        :param blur_limit:
        :param sigmaX_limit:
        :param sigmaY_limit:
        :param rotate_limit:
        :param beta_limit:
        :param noise_limit:
        :param p:
        :return:
        """
        aug = A.AdvancedBlur(blur_limit=blur_limit, sigmaX_limit=sigmaX_limit, sigmaY_limit=sigmaY_limit, rotate_limit=rotate_limit, beta_limit=beta_limit, noise_limit=noise_limit, p=p)
        return aug(image=image)['image']

    def Blur(image, ksize=(10,10), p=1):
        """
        :param image:
        :param ksize:
        :return:
        """
        if random.random() < p:
            return cv2.blur(image, ksize=ksize)
        else:
            return  image

    def GaussianBlur(self, image, ksize=(11, 11), sigmaX=0, sigmaY=0, p=1):
        """
        :param image:
        :param ksize: lưu ý phải là số lẽ
        :param sigmaX: Độ lệnh chuẩn --> Thể hiện mức độ làm mờ lệch về phía phải (trục x)
        :param sigmaY: Thể hiện mức độ làm mờ lệch xuống (trục y)
        :return:
        """
        if random.random() < p:
            return cv2.GaussianBlur(image, ksize=ksize, sigmaX=sigmaX, sigmaY=sigmaY)
        else:
            return image

    def Noise(self, image, var_limit=(500.0, 1000.0), mean=0, per_channel=True, p=1):
        """
        :param image:
        :param var_limit: phương sai của nhiễu tỉ lệ thuận với độ lớn của nhiễu
        :param mean:
        :param per_channel: Nếu True --> sẽ lấy nhiễu lần lượt cho các kênh, ngược lại thì sẽ áp dụng một phân bố nhiễu cho cả 3 kênh
        :param p:
        :return:
        """
        aug = A.GaussNoise(var_limit=var_limit, mean=mean, per_channel=per_channel, p=p)
        return aug(image=image)['image']

    def GrayScale(self, image, p=1):
        if random.random() < p:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # -----------------------------------------------------------------
    # Spatial-level transforms
    def RandomRotate90(self, image, mask, p=1):
        """
        Cái này là xoay cả bức ảnh theo 4 góc 90, 180, 270, 360 độ
        :param image:
        :param mask:
        :param p:
        :return:
        """
        aug = A.RandomRotate90(p=1)
        results = aug(image=image, mask=mask)
        return results['image'], results['mask']

    def Rotate(self, image, mask, limit = -180, interpolation = 1, border_mode = 4, value = None, mask_value = None, method = 'largest_box', crop_border = False, p = 1):
        """
        :param image:
        :param limit: Góc xoay, kiểu xoay ảnh xong padding các phần thiếu bằng cách nhân phần ảnh chính lên, Nếu muốn phần padding là nền đen thì set border_mode=0, value=0 (trắng thì 255)
        :param interpolation: index của [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
        :param border_mode:  [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101]
        :param value:
        :param mask_value:
        :param method:
        :param crop_border: Nếu set True --> Không cần padding thêm mà nó sẽ phóng to hình ảnh lên để xoay --> Làm thay đổi kích thước của ảnh
        :return:
        """
        aug = A.Rotate(limit = limit, interpolation = interpolation, border_mode = border_mode, value = value, mask_value = mask_value, method = method, crop_border = crop_border, p = p)
        results = aug(image=image, mask=mask)
        return results['image'], results['mask']

    def Perspective(self, image, mask, scale=(0.05, 0.2), keep_size=True, pad_mode=0, pad_val=0, mask_pad_val=0, fit_output=False, interpolation=1, p=1):
        """
        :param scale: Độ biến dạng của ảnh
        :param keep_size:
        :param pad_mode: OpenCV border mode.
        :param pad_val: padding value if border_mode is cv2.BORDER_CONSTANT. Default: 0
        :param mask_pad_val:
        :param fit_output:
        :param interpolation:
        :return:
        """
        aug = A.Perspective(scale=scale, keep_size=keep_size, pad_mode=pad_mode, pad_val=pad_val, mask_pad_val=mask_pad_val, fit_output=fit_output, interpolation=interpolation, p=p)
        results = aug(image=image, mask=mask)
        return results['image'], results['mask']


    # def RandomCropAndResize(self, image, mask, x=0.8, w2h_ratio=1.0, interpolation=1, p=1):
    #     """
    #     :param image:
    #     :param mask:
    #     :param x: càng nhỏ thì kích thước vùng cắt càng nhỏ
    #     :param w2h_ratio:
    #     :param interpolation:
    #     :param p:
    #     :return:
    #     """
    #     original_height, original_width = image.shape[:2]
    #
    #     min = original_height if (original_height - original_width) < 0 else original_width
    #
    #     if x < (0.3 * min):
    #         """
    #             Nếu x quá nhỏ thì vùng crop ra sẽ rất nhỏ --> nhiều khi crop ra vùng không có kích thước
    #         """
    #         x = 0.3
    #
    #     aug = A.RandomSizedCrop(min_max_height=(int(min * x), int(min * x)), height=original_height, width=original_width, w2h_ratio=w2h_ratio, interpolation=interpolation, p=p)
    #     results = aug(image=image, mask=mask)
    #     return results['image'], results['mask']