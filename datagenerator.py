import numpy as np
import random
import cv2

class Label:
    none = 0
    centerline = 1
    water = 2

class DataGenerator():
    """
    Generates synthetic centerline data
    """

    def __init__(self, image_size=(1024, 1024)):
        self.max_branches = 3
        self.image_size = image_size

        # list of distortion functions to pick from
        self.dist_func = [self.random_gaussian_blur,
                          self.random_brightness_contrast,
                          self.random_noise]

    def __iter__(self):
        return self

    def __next__(self):
        # outputs: labels and width map
        labels = np.zeros(self.image_size, dtype=np.uint8)
        width_map = np.zeros(self.image_size, dtype=np.uint8)

        R, C = self.image_size

        # draw the main channel
        width = random.randint(1, 255)
        p1_x = random.randint(width, C-width)
        p2_x = random.randint(width, C-width)
        p1_y = 0
        p2_y = R-1
        cv2.line(labels, (p1_x, p1_y), (p2_x, p2_y), color=(Label.centerline), thickness=1)
        cv2.line(width_map, (p1_x, p1_y), (p2_x, p2_y), color=(width), thickness=width)

        # draw branches
        width_b = width
        num_branches = random.randint(0, self.max_branches)
        for i in range(num_branches):
            [row,col] = np.where(labels)

            idx = random.randint(0, len(row)-1)
            p3_x = col[idx]
            p3_y = row[idx]

            p4_x = random.choice([0, C-1])
            p4_y = random.randint(0, R-1)
            width_b = random.randint(1, width_b)

            cv2.line(labels, (p3_x, p3_y), (p4_x, p4_y), color=(Label.centerline), thickness=1)

            width_map_b = np.zeros(self.image_size)
            cv2.line(width_map_b, (p3_x, p3_y), (p4_x, p4_y), color=(width_b), thickness=width_b)

            # merge width maps
            idx = width_map_b > width_map
            width_map[idx] = width_map_b[idx]

        # generate the shoreline
        if random.random() > 0.75:
            p5_y = random.randint(width, R-width)
            p6_y = random.randint(width, R-width)

            p5_x = 0
            p6_x = C-1

            cv2.line(labels, (p5_x, p5_y), (p6_x, p6_y), color=(Label.water), thickness=1)
            [row,col] = np.where(labels == Label.water)
            for j in range(len(row)):
                width_map[row[j]:, col[j]] = 0
                labels[row[j]:, col[j]] = Label.water

        # transpose randomly
        if random.random() > 0.5:
            labels = labels.T
            width_map = width_map.T

        [row,col] = np.where((width_map > 0) & (labels == 0))
        labels[row, col] = Label.water

        # create the input mask
        input_mask = (labels > 0).astype(np.float32)

        # apply distortions to simulate real imaging conditions
        random.shuffle(self.dist_func)
        for dist_func in self.dist_func:
            if random.random() < 0.5:
                input_mask = dist_func(input_mask)

        return input_mask, labels, width_map

    def random_gaussian_blur(self, image):
        k_size = random.randrange(3, 35, 2)
        image = cv2.GaussianBlur(image, (k_size, k_size), 0)
        image = np.minimum(np.maximum(image, 0), 1)
        return image

    def random_brightness_contrast(self, image):
        brightness = 0.25 + random.random() / 2
        contrast = 0.25 + random.random() / 2
        image = contrast * (image - np.mean(image)) / np.std(image) + brightness
        image = np.minimum(np.maximum(image, 0), 1)
        return image

    def random_noise(self, image):
        noise_var = random.random() / 20
        noise = np.random.randn(image.shape[0], image.shape[1]) * noise_var
        image += noise
        image = np.minimum(np.maximum(image, 0), 1)
        return image

if __name__ == '__main__':
    dg = DataGenerator()
    input_mask, labels, width_map = next(dg)
    cv2.imwrite("centerlines.png", labels * 127)
    cv2.imwrite("mask.png", (input_mask * 255).astype(np.uint8))
    cv2.imwrite("width_map.png", width_map)