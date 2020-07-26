#!/usr/bin/python

import CustomeDataloader
ds = CustomeDataloader.ImagenetDatasets("/data/datasets/imagenets/raw",True)
transforms = CustomeDataloader.Transforms(True,False,False,3,224,224,0.5)
dl = CustomeDataloader.Dataloader(ds,transforms,150,4,8,8,16,48,100,True,True,False)
dl_len = len(dl)
for data in range(0,dl_len):
    data = dl.next()
    print (type(data),len(data),type(data[0]),data[0].shape, data[1].shape)

