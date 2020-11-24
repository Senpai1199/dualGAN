import numpy as np
import os
import glob
from PIL import Image
from scipy.misc import imresize, toimage


def preprocess_for_saving_image(im):
    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)

    # Scale to 0-255
    #im = (((im - im.min()) * 255) / (im.max() - im.min())).astype(np.uint8)
    im = ((im + 1.0) * 127.5).astype(np.uint8)

    return im

def save_result(image_fn,
                real_image_u, g_image_u_to_v, g_image_u_to_v_to_u,
                real_image_v, g_image_v_to_u, g_image_v_to_u_to_v):
    im_0 = preprocess_for_saving_image(real_image_u)
    im_1 = preprocess_for_saving_image(g_image_u_to_v)
    im_2 = preprocess_for_saving_image(g_image_u_to_v_to_u)
    im_3 = preprocess_for_saving_image(real_image_v)
    im_4 = preprocess_for_saving_image(g_image_v_to_u)
    im_5 = preprocess_for_saving_image(g_image_v_to_u_to_v)

    concat_0 = np.concatenate((im_0, im_1, im_2), axis=1)
    concat_1 = np.concatenate((im_3, im_4, im_5), axis=1)
    concated = np.concatenate((concat_0, concat_1), axis=1)

    if concated.shape[2] == 1:
        reshaped = np.squeeze(concated, axis=2)
        toimage(reshaped, mode='L').save(image_fn)
    else:
        toimage(concated, mode='RGB').save(image_fn)

def save_result_single_row(image_fn, real_image_u, g_image_one_path, g_image_two_path):
    im_0 = preprocess_for_saving_image(real_image_u)
    im_1 = preprocess_for_saving_image(g_image_one_path)
    im_2 = preprocess_for_saving_image(g_image_two_path)
    concated = np.concatenate((im_0, im_1, im_2), axis=1)

    if concated.shape[2] == 1:
        reshaped = np.squeeze(concated, axis=2)
        toimage(reshaped, mode='L').save(image_fn)
    else:
        toimage(concated, mode='RGB').save(image_fn)



# class for loading images
class Dataset(object):
    def __init__(self, input_dir_u, input_dir_v, fn_ext, im_size, im_channel_u, im_channel_v, do_flip, do_shuffle):
        if not os.path.exists(input_dir_u) or not os.path.exists(input_dir_v):
            raise Exception('input directory does not exists!!')

        # search for images
        if 'train' in input_dir_u:
            self.image_files_u = ['A_1014.jpg', 'A_111.jpg', 'A_1113.jpg', 'A_1193.jpg', 'A_1301.jpg', 'A_1367.jpg', 'A_1457.jpg', 'A_1565.jpg', 'A_1628.jpg', 'A_1731.jpg', 'A_1808.jpg', 'A_1913.jpg', 'A_2015.jpg', 'A_2091.jpg', 'A_2160.jpg', 'A_2240.jpg', 'A_229.jpg', 'A_2352.jpg', 'A_2452.jpg', 'A_2570.jpg', 'A_2689.jpg', 'A_2787.jpg', 'A_2905.jpg', 'A_2964.jpg', 'A_302.jpg', 'A_3052.jpg', 'A_3132.jpg', 'A_3223.jpg', 'A_3301.jpg', 'A_3381.jpg', 'A_3464.jpg', 'A_3552.jpg', 'A_3637.jpg', 'A_3700.jpg', 'A_3761.jpg', 'A_381.jpg', 'A_3821.jpg', 'A_3901.jpg', 'A_4002.jpg', 'A_4066.jpg', 'A_4166.jpg', 'A_4266.jpg', 'A_4356.jpg', 'A_4436.jpg', 'A_4496.jpg', 'A_452.jpg', 'A_4596.jpg', 'A_4696.jpg', 'A_4776.jpg', 'A_4866.jpg', 'A_4937.jpg', 'A_5026.jpg', 'A_5090.jpg', 
    'A_5150.jpg', 'A_521.jpg', 'A_5233.jpg', 'A_5322.jpg', 'A_5399.jpg', 'A_5479.jpg', 'A_5571.jpg', 'A_5656.jpg', 'A_5768.jpg', 'A_585.jpg', 'A_5872.jpg', 'A_5974.jpg', 'A_6023.jpg', 'A_6088.jpg', 'A_6163.jpg', 'A_6223.jpg', 'A_6305.jpg', 'A_6368.jpg', 'A_6438.jpg', 'A_648.jpg', 'A_6518.jpg', 'A_6568.jpg', 'A_6628.jpg', 'A_6728.jpg', 'A_6819.jpg', 'A_6901.jpg', 'A_6961.jpg', 'A_7024.jpg', 'A_7146.jpg', 'A_7226.jpg', 'A_7326.jpg', 'A_7426.jpg', 'A_7546.jpg', 'A_755.jpg', 'A_7627.jpg', 'A_834.jpg', 'A_934.jpg']
            self.image_files_v = ['B_1014.jpg', 'B_111.jpg', 'B_1113.jpg', 'B_1193.jpg', 'B_1301.jpg', 'B_1367.jpg', 'B_1457.jpg', 'B_1565.jpg', 'B_1628.jpg', 'B_1731.jpg', 'B_1808.jpg', 'B_1913.jpg', 'B_2015.jpg', 'B_2091.jpg', 'B_2160.jpg', 'B_2240.jpg', 'B_229.jpg', 'B_2352.jpg', 'B_2452.jpg', 'B_2570.jpg', 'B_2689.jpg', 'B_2787.jpg', 'B_2905.jpg', 'B_2964.jpg', 'B_302.jpg', 'B_3052.jpg', 'B_3132.jpg', 'B_3223.jpg', 'B_3301.jpg', 'B_3381.jpg', 'B_3464.jpg', 'B_3552.jpg', 'B_3637.jpg', 'B_3700.jpg', 'B_3761.jpg', 'B_381.jpg', 'B_3821.jpg', 'B_3901.jpg', 'B_4002.jpg', 'B_4066.jpg', 'B_4166.jpg', 'B_4266.jpg', 'B_4356.jpg', 'B_4436.jpg', 'B_4496.jpg', 'B_452.jpg', 'B_4596.jpg', 'B_4696.jpg', 'B_4776.jpg', 'B_4866.jpg', 'B_4937.jpg', 'B_5026.jpg', 'B_5090.jpg', 
    'B_5150.jpg', 'B_521.jpg', 'B_5233.jpg', 'B_5322.jpg', 'B_5399.jpg', 'B_5479.jpg', 'B_5571.jpg', 'B_5656.jpg', 'B_5768.jpg', 'B_585.jpg', 'B_5872.jpg', 'B_5974.jpg', 'B_6023.jpg', 'B_6088.jpg', 'B_6163.jpg', 'B_6223.jpg', 'B_6305.jpg', 'B_6368.jpg', 'B_6438.jpg', 'B_648.jpg', 'B_6518.jpg', 'B_6568.jpg', 'B_6628.jpg', 'B_6728.jpg', 'B_6819.jpg', 'B_6901.jpg', 'B_6961.jpg', 'B_7024.jpg', 'B_7146.jpg', 'B_7226.jpg', 'B_7326.jpg', 'B_7426.jpg', 'B_7546.jpg', 'B_755.jpg', 'B_7627.jpg', 'B_834.jpg', 'B_934.jpg']
        else:
            self.image_files_u = ['A_7722.jpg', 'A_7820.jpg', 'A_7897.jpg', 'A_7976.jpg', 'A_8056.jpg', 'A_8132.jpg', 'A_8224.jpg', 'A_8288.jpg', 'A_8395.jpg', 'A_8493.jpg']
            self.image_files_v = ['B_7722.jpg', 'B_7820.jpg', 'B_7897.jpg', 'B_7976.jpg', 'B_8056.jpg', 'B_8132.jpg', 'B_8224.jpg', 'B_8288.jpg', 'B_8395.jpg', 'B_8493.jpg']
        
        for i, path in enumerate(self.image_files_u):
            self.image_files_u[i] = input_dir_u + self.image_files_u[i]

        for i, path in enumerate(self.image_files_v):
            self.image_files_v[i] = input_dir_v + self.image_files_v[i]


        if len(self.image_files_u) == 0 or len(self.image_files_v) == 0:
            raise Exception('input directory does not contain any images!!')

        # shuffle image files
        self.do_shuffle = do_shuffle
        if self.do_shuffle:
            np.random.shuffle(self.image_files_u)
            np.random.shuffle(self.image_files_v)

        self.n_images = len(self.image_files_u) if len(self.image_files_u) <= len(self.image_files_v) else len(self.image_files_v)
        self.batch_index = 0
        self.resize_to = im_size
        self.color_mode_u = 'L' if im_channel_u == 1 else 'RGB'
        self.color_mode_v = 'L' if im_channel_v == 1 else 'RGB'
        self.do_flip = do_flip
        self.image_max_value = 255
        self.image_max_value_half = 127.5
        # self.prng = np.random.RandomState(777)

    def reset(self):
        self.batch_index = 0
        # shuffle image files
        if self.do_shuffle:
            np.random.shuffle(self.image_files_u)
            np.random.shuffle(self.image_files_v)

    def get_image_by_index(self, index):
        if index >= self.n_images:
            index = 0

        fn_u = [self.image_files_u[index]]
        fn_v = [self.image_files_v[index]]
        image_u = self.load_image(fn_u, self.color_mode_u)
        image_v = self.load_image(fn_v, self.color_mode_v)
        return image_u, image_v

    def get_image_by_index_u(self, index):
        if index >= self.n_images:
            index = 0

        fn_u = [self.image_files_u[index]]
        image_u = self.load_image(fn_u, self.color_mode_u)
        return image_u

    def get_image_by_index_v(self, index):
        if index >= self.n_images:
            index = 0

        fn_v = [self.image_files_v[index]]
        image_v = self.load_image(fn_v, self.color_mode_v)
        return image_v

    def get_next_batch(self, batch_size):
        if (self.batch_index + batch_size) > self.n_images:
            self.batch_index = 0

        batch_files_u = self.image_files_u[self.batch_index:self.batch_index + batch_size]
        batch_files_v = self.image_files_v[self.batch_index:self.batch_index + batch_size]

        images_u = self.load_image(batch_files_u, self.color_mode_u)
        images_v = self.load_image(batch_files_v, self.color_mode_v)

        self.batch_index += batch_size

        return images_u, images_v

    def load_image(self, fn_list, color_mode):
        images = []
        for fn in fn_list:
            # open images with PIL
            im = Image.open(fn)
            im = np.array(im.convert(color_mode))

            # resize
            im = imresize(im, [self.resize_to, self.resize_to])

            # perform flip if needed
            # random_val = self.prng.uniform(0, 1)
            random_val = np.random.random()
            if self.do_flip and random_val > 0.5:
                #im = np.flip(im, axis=1)
                im = np.fliplr(im)

            # normalize input [0 ~ 255] ==> [-1 ~ 1]
            #im = (im / self.image_max_value - 0.5) * 2
            im = im / self.image_max_value_half - 1.0

            # make 3 dimensional for single channel image
            if len(im.shape) < 3:
                im = np.expand_dims(im, axis=2)

            images.append(im)
        images = np.array(images)

        return images