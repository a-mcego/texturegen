import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import time

import sys
IS_COLAB = 'google.colab' in sys.modules

print(f"IS_COLAB: {IS_COLAB}")

OUTPUT_SHAPE = [512,512]
PATCH_SHAPE = [16,16]
BATCH_SIZE = 64
STACKING_SIZE = 2
LEARNING_RATE_D = 0.004
LEARNING_RATE_G = 0.001
SAVE_INTERVAL = 1024
#SRC_IMAGE = "sky.png"
#SRC_IMAGE = "gravel.png"
#SRC_IMAGE = "grassflower.png"
SRC_IMAGE = "ff6.png"
PRINT_TIME = 5000


PATCH_SHAPE = tf.convert_to_tensor(PATCH_SHAPE)

def img_int8tofloat(x):
    return tf.cast(x,tf.float32)/255.0*2.0-1.0

if IS_COLAB:
    from google.colab import drive
    drive.mount('/content/gdrive')
    imgfilename = f"/content/gdrive/My Drive/texgen/input/{SRC_IMAGE}"
else:
    imgfilename = f"inputs\\{SRC_IMAGE}"

real_img = img_int8tofloat(tf.io.decode_image(tf.io.read_file(imgfilename)))
real_img = real_img[None]
print(real_img.shape, real_img.dtype)

@tf.function(jit_compile=False)
def realimg():
    pshape = PATCH_SHAPE
    
    multiplier = tf.random.uniform([], minval=1, maxval=5, dtype=tf.dtypes.int32)
    
    ys = tf.random.uniform([BATCH_SIZE,STACKING_SIZE], minval=0, maxval=1000000000, dtype=tf.dtypes.int32)
    xs = tf.random.uniform([BATCH_SIZE,STACKING_SIZE], minval=0, maxval=1000000000, dtype=tf.dtypes.int32)
    
    out = []
    for b_i in range(BATCH_SIZE):
        stack = []
        for s_i in range(STACKING_SIZE):
            actualp = pshape
            y = ys[b_i,s_i]%(real_img.shape[2]-actualp[0])
            x = xs[b_i,s_i]%(real_img.shape[3]-actualp[1])
            patch = real_img[
                :,
                :,
                y:y+actualp[0], 
                x:x+actualp[1]
            ]
            stack.append(patch)
        out.append(tf.concat(stack,axis=-1))
    
    ret = tf.concat(out,axis=0)
    return ret
    

class FakeImg(tf.keras.Model):
    def __init__(self):
        super(FakeImg,self).__init__()
        initer = tf.zeros_initializer()
        #initer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
    
        self.img = self.add_weight('img',shape=[1,OUTPUT_SHAPE[0],OUTPUT_SHAPE[1],3], initializer=initer, trainable=True)
        
    def build(self, shape):
        pass
        #initer = tf.zeros_initializer()
        #initer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
    
        #self.img = self.add_weight('img',shape=[1,OUTPUT_SHAPE[0],OUTPUT_SHAPE[1],3], initializer=initer, trainable=True)

    @tf.function(jit_compile=False)
    def call(self, _):
        processed_img = self.img
        processed_img = tf.reshape(processed_img, [1,OUTPUT_SHAPE[0],OUTPUT_SHAPE[1],3])
        processed_img = tf.concat([processed_img, processed_img[:,:PATCH_SHAPE[0]-1]], axis=-3)
        processed_img = tf.concat([processed_img, processed_img[:,:,:PATCH_SHAPE[1]-1]], axis=-2)
    
        ys = tf.random.uniform([BATCH_SIZE*STACKING_SIZE], minval=0, maxval=OUTPUT_SHAPE[0], dtype=tf.dtypes.int32)
        xs = tf.random.uniform([BATCH_SIZE*STACKING_SIZE], minval=0, maxval=OUTPUT_SHAPE[1], dtype=tf.dtypes.int32)
        
        out = []
        for b_i in range(BATCH_SIZE):
            stack = []
            for s_i in range(STACKING_SIZE):
                bs_i = b_i*STACKING_SIZE+s_i
                patch = processed_img[:,ys[bs_i]:ys[bs_i]+PATCH_SHAPE[0], xs[bs_i]:xs[bs_i]+PATCH_SHAPE[1]]
                patch = tf.reshape(patch, [1, PATCH_SHAPE[0], PATCH_SHAPE[1], 3])
                stack.append(patch)
            out.append(tf.concat(stack,axis=-1))
        
        ret = tf.concat(out,axis=0)
        ret = tf.reshape(ret, [BATCH_SIZE, PATCH_SHAPE[0], PATCH_SHAPE[1], STACKING_SIZE*3])
        #print(f"returning fake: {ret.shape}")
        return ret
        
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator,self).__init__()
        
        self.convs = []
        self.convs.append(tf.keras.layers.Conv2D(filters=24*4, kernel_size=3, activation=tf.nn.relu, padding="same"))
        self.convs.append(tf.keras.layers.Conv2D(filters=32*4, kernel_size=3, activation=tf.nn.relu, padding="same"))
        self.convs.append(tf.keras.layers.Conv2D(filters=64*4, kernel_size=3, activation=tf.nn.relu, padding="same"))

        self.convs2 = []
        self.convs2.append(tf.keras.layers.Conv2D(filters=24*4, kernel_size=3, activation=tf.nn.relu, padding="same"))
        self.convs2.append(tf.keras.layers.Conv2D(filters=32*4, kernel_size=3, activation=tf.nn.relu, padding="same"))
        self.convs2.append(tf.keras.layers.Conv2D(filters=64*4, kernel_size=3, activation=tf.nn.relu, padding="same"))

        self.lns = [tf.keras.layers.LayerNormalization(axis=-1) for _ in range(3)]
        self.lns2 = [tf.keras.layers.LayerNormalization(axis=-1) for _ in range(3)]
        
        self.pools = [tf.keras.layers.AveragePooling2D() for _ in range(2)]
        self.pools.append(None)

        self.lastdense = tf.keras.layers.Dense(1, use_bias=False)
                
    @tf.function(jit_compile=True)
    def call(self, inputdata):
        for n in range(3):
            inputdata = self.convs[n](inputdata)
            inputdata = inputdata + self.convs2[n](self.lns2[n](inputdata))
            inputdata = self.lns[n](inputdata)
            if self.pools[n] is not None:
                inputdata = self.pools[n](inputdata)
            
        inputdata = tf.reshape(inputdata, [inputdata.shape[0], -1])
        inputdata = self.lastdense(inputdata)
        inputdata = tf.squeeze(inputdata, axis=-1)
        return inputdata

fakeimg = FakeImg()
d = Discriminator()

optimizer_d = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_D, amsgrad=True)
optimizer_g = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_G, amsgrad=True)

iters = 0

@tf.function(jit_compile=False)
def do_thing():
    fakes = d(fakeimg(1))
    reals = d(realimg())
    
    reals = reals[None,:]
    fakes = fakes[:,None]
    return fakes-reals
        

@tf.function(jit_compile=False)
def train_D():
    #train discriminator
    with tf.GradientTape() as tape:
        loss = tf.math.softplus(do_thing())
        
    gradients = tape.gradient(loss, d.trainable_variables)
    optimizer_d.apply_gradients(zip(gradients, d.trainable_variables))
    
@tf.function(jit_compile=False)
def train_G():    
    #train generator
    with tf.GradientTape() as tape:
        loss = tf.nn.relu(-do_thing())
        
    gradients = tape.gradient(loss, fakeimg.trainable_variables)
    optimizer_g.apply_gradients(zip(gradients, fakeimg.trainable_variables))
    
    
currtime = time.time()
curriters = 0
while True:
    iters += 1
    curriters += 1
    train_D()
    if iters >= 64:
        train_G()

    if (time.time()-currtime)*1000.0 > PRINT_TIME:
        delta = time.time()-currtime
        
        print(f"#{iters}, {delta*1000.0/curriters} ms/iter")

        currtime = time.time()
        curriters = 0
    
    #print(f"{iters}",end="    \r")asd asd asd asd 
    if iters%SAVE_INTERVAL == 0:
        img = (tf.squeeze(fakeimg.img, axis=0)+1.0)*127.5
        img = tf.clip_by_value(img, 0.0, 255.0)
        img = tf.cast(img, tf.dtypes.uint8)
        
        if IS_COLAB:
            tf.io.write_file(f"/content/gdrive/My Drive/texgen/{iters}.png", tf.io.encode_png(img))
        else:
            tf.io.write_file(f"next_{iters}.png", tf.io.encode_png(img))
        
    
    #print(tf.reduce_sum(fakeimg.img))