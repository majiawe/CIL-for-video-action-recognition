import scipy.io
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

root = '/data1/ma_gps/Weizmann_Dataset/aligned_masks/'

data = scipy.io.loadmat('/data1/ma_gps/Weizmann_Dataset/classification_masks.mat')
aligned_masks = data['aligned_masks']
name = ['daria','denis','eli','ido','ira','lena','lyova','moshe','shahar']
cls = ['bend','jack','jump','pjump','run','side','skip','walk','wave1','wave2']
cls_ad = ['skip1','skip2','walk1','walk2','run1','run2']
naa = ['lena']
# for cl in cls:
#     # clas = cl[:-1]
#     if not os.path.exists(os.path.join(root, cl)):
#         os.makedirs(os.path.join(root, cl))
#     for na in name:
#         if not os.path.exists(os.path.join(root, cl,na +'_'+ cl)):
#             os.makedirs(os.path.join(root, cl,na +'_'+ cl))
#         try:
#             video = aligned_masks[0][0][na+'_'+cl]
#         except:
#             continue
#         video = video.transpose(2, 0, 1)
#         for j,img in enumerate(video):
#
#             im = Image.fromarray(img)
#
#
#             plt.imsave(os.path.join(root, cl,na +'_'+ cl, 'img_{:05d}.jpg'.format(j+1)),im,cmap='gray')





# video = aligned_mask[0][0]['daria_bend']
#
# print(np.shape(video))
# video = video.transpose(2,0,1)
# print(video)
#
# for i in range(20,80,4):
#     im = Image.fromarray(video[i])
#     plt.imshow(im,cmap='gray')
#     plt.show()
