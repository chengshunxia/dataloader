#!/usr/bin/python

import CustomeDataloader
import time
ds = CustomeDataloader.ImagenetDatasets("/mnt/data/dataset/imagenet/raw.bak/",True)
transforms = CustomeDataloader.Transforms(True,False,False,3,224,224,0.5)
dl = CustomeDataloader.Dataloader(ds,transforms,150,4096,48,100,True,True,False)
time.sleep(30)
dl_len = len(dl)
for data in range(0,dl_len):
    data = dl.next()
    print (data.image(), data.label())
#    print (type(data),len(data),type(data[0]),data[0].shape, data[1].shape)

