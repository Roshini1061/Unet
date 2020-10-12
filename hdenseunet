import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.optimizers import *
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import *

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
	x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
	x = BatchNormalization(axis=3, scale=False)(x)
	if(activation == None):
		return x
	x = Activation(activation, name=name)(x)
	return x
def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
	x = Conv2DTranspose(filters, (num_row, num_col), strides=strides, padding=padding)(x)
	x = BatchNormalization(axis=3, scale=False)(x)
	return x


def MultiResBlock(U, inp, alpha = 1.67):
	W = alpha * U
	shortcut = inp
	shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) + int(W*0.5), 1, 1, activation=None, padding='same')
	conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3, activation='relu', padding='same')
	conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3, activation='relu', padding='same')
	conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3, activation='relu', padding='same')
	out = concatenate([conv3x3, conv5x5, conv7x7], axis=3)
	out = BatchNormalization(axis=3)(out)
	out = add([shortcut, out])
	out = Activation('relu')(out)
	out = BatchNormalization(axis=3)(out)
	return out


def ResPath(filters, length, inp):
	shortcut = inp
	shortcut = conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')
	out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')
	out = add([shortcut, out])
	out = Activation('relu')(out)
	out = BatchNormalization(axis=3)(out)
	for i in range(length-1):
		shortcut = out
		shortcut = conv2d_bn(shortcut, filters, 1, 1, activation=None, padding='same')
		out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')
		out = add([shortcut, out])
		out = Activation('relu')(out)
		out = BatchNormalization(axis=3)(out)
	return out

   
class mymultiresUnet(object):
	def __init__(self, img_rows = 512, img_cols = 512):
		self.img_rows = img_rows
		self.img_cols = img_cols
# 参数初始化定义
	def load_data(self):
		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train, imgs_test
# 载入数据
	def get_multiresunet(self):
		inputs = Input((self.img_rows, self.img_cols,1))
		mresblock1 = MultiResBlock(32, inputs)
		pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
		mresblock1 = ResPath(32, 4, mresblock1)
		mresblock2 = MultiResBlock(32*2, pool1)
		pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
		mresblock2 = ResPath(32*2, 3, mresblock2)
		mresblock3 = MultiResBlock(32*4, pool2)
		pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
		mresblock3 = ResPath(32*4, 2, mresblock3)
		mresblock4 = MultiResBlock(32*8, pool3)
		pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
		mresblock4 = ResPath(32*8, 1, mresblock4)
		mresblock5 = MultiResBlock(32*16, pool4)
		up6 = concatenate([Conv2DTranspose(32*8, (2, 2), strides=(2, 2), padding='same')(mresblock5), mresblock4], axis=3)
		mresblock6 = MultiResBlock(32*8, up6)
		up7 = concatenate([Conv2DTranspose(32*4, (2, 2), strides=(2, 2), padding='same')(mresblock6), mresblock3], axis=3)
		mresblock7 = MultiResBlock(32*4, up7)
		up8 = concatenate([Conv2DTranspose(32*2, (2, 2), strides=(2, 2), padding='same')(mresblock7), mresblock2], axis=3)
		mresblock8 = MultiResBlock(32*2, up8)
		up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(mresblock8), mresblock1], axis=3)
		mresblock9 = MultiResBlock(32, up9)
		conv10 = conv2d_bn(mresblock9, 1, 1, 1, activation='sigmoid')

		model = Model(inputs = inputs, outputs = conv10)
		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
		return model


# 如果需要修改输入的格式，那么可以从以下开始修改，上面的结构部分不需要修改
	def train(self):
		print("loading data")
		imgs_train, imgs_mask_train, imgs_test = self.load_data()
		print("loading data done")
		model = self.get_unet()
		print("got unet")
		model_checkpoint = ModelCheckpoint('my_unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=2, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])
		print('predict test data')
		imgs_mask_test = model.predict(imgs_test, batch_size=2, verbose=1)
		np.save('/content/drive/My Drive/Unet-master/results/imgs_mask_test.npy', imgs_mask_test)

	def save_img(self):
		print("array to image")
		imgs = np.load('/content/drive/My Drive/Unet-master/results/imgs_mask_test.npy')
		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save("/content/drive/My Drive/Unet-master/results/%d.jpg"%(i+1))

if __name__ == '__main__':
	myunet = mymultiresUnet()
	myunet.train()
	myunet.save_img()
