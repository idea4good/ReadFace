import tensorflow as tf
import numpy as np
import inception_resnet_v1 as resnet

class c_face_attr_reader(object):
    def __init__(self, face_img_size, model_path = 'models/model-20170512-110547.ckpt-250000'):
        self.face_imgs = tf.placeholder('float', [None, face_img_size, face_img_size, 3])
        self.face_img_size = face_img_size
        self.attr = tf.nn.l2_normalize(resnet.inference(self.face_imgs, 0.6, phase_train=False)[0], 1, 1e-10); #some magic numbers that u dont have to care about
        self.sess = tf.Session()
        saver = tf.train.Saver() #saver load pretrain model
        saver.restore(self.sess, model_path)
        print('Moldes loaded')

    def get_face_attr(self, face_imgs):
        return self.sess.run(self.attr, feed_dict = {self.face_imgs : self.load_data_list(face_imgs, self.face_img_size)})

    def prewhiten(self, img):
        mean = np.mean(img)
        std = np.std(img)
        std_adj = np.maximum(std, 1.0 / np.sqrt(img.size))
        return np.multiply(np.subtract(img, mean), 1 / std_adj)

    def load_data_list(self, imgList, image_size, do_prewhiten=True):
        images = np.zeros((len(imgList), image_size, image_size, 3))
        i = 0
        for img in imgList:
            if img is not None:
                if do_prewhiten:
                    img = self.prewhiten(img)
                images[i, :, :, :] = img
                i += 1
        return images