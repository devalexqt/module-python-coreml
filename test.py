import sys
import time
import cv2
import numpy as np
import mycoreml
print(">>>started, python version: %s, time: %.3f"%(sys.version,time.time()))

status = mycoreml.test("ls -l")

print(">>STATUS:",status)

status2 = mycoreml.test2("ls -l")

print(">>STATUS2:",status2)

mycoreml.load("/Users/hetzner/Documents/test-coreml/ModelCoreMl_autoencoder_compiled/ModelCoreMl_autoencoder.mlmodelc","ALL")#precompiled model
# mycoreml.load("/Users/hetzner/Documents/test-coreml/ModelCoreMl_autoencoder.mlmodel","ALL")

# img = ((np.random.rand(3,256,256)*255)/255).clip(0,255).astype(np.float32)
img=cv2.imread("/Users/hetzner/Movies/test.png")
img=cv2.resize(img,(256,256))
img=np.transpose(img,(2,0,1))
img=img/255
img=img.astype(np.float32)
img=np.expand_dims(img,0)
# print(">>>input img:",img.shape,img)
img_buff=((np.random.rand(4021,4021,3)*255)).astype(np.uint8)
result=[]
for i in range(20):
    start_time=time.time()
    result=mycoreml.predict(img,0,1)
    print("predict result shape:",result.shape,result.dtype)
    # print(result)
    # result=np.squeeze(result)
    # result=np.transpose(result,(1,2,0))
    # result=result*255
    # result=result.clip(0,255).astype(np.uint8)#2-3ms
    # result=result.astype(np.uint8)
    # img_buff[1024:1024+944,1024:1024+944]=result[40:40+944,40:40+944,]
    print(">>predict time: %.3fs:"%(time.time()-start_time))
# print(">>result after:",result.shape, result)

cv2.imwrite("/Users/hetzner/Movies/out_python_custom_module.png", result)