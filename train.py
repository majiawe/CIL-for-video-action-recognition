from DataLoader import dataLoader,DataProcess
from gwr_tool import episodic_gwr,agwr
import numpy as np


# kth = '/data1/ma_gps/KTH_dataset/'
weizmann = '/data1/ma_gps/Weizmann_Dataset/'
data = DataProcess(weizmann)

#train_dataset
ds_frame = dataLoader(train=True,frames=True,mask_frame='aligned_masks')


#test_dataset
frame_test = dataLoader(train=False,frames=True,mask_frame='aligned_masks')


e_labels = [10]  #6 for KTH  and  10 for weizam
P1_net = episodic_gwr.EpisodicGWR()

# initiate Posture network 1 and train it
P1_net.init_network(ds=ds_frame,dimension=512,e_labels=e_labels,num_context=3)


all_th = np.array([0.009,0.009,0.009,0.009,0.009,0.009,0.009,0.009,0.009,0.009])
train_cls = ['bend','skip','wave1','jack','side','wave2','pjump','run','jump','walk']
#train_cls = ['handwaving','boxing','handclapping','walking','jogging','running']  #KTH
trained_cls = []

train_weights = []
weights_label = []
for i,cls in enumerate(train_cls):
    trained_cls.append(cls)
    data.train_test_set(ls_cls=cls)
    train_frame = dataLoader(train=True,frames=True,mask_frame='aligned_masks')
    #train_p1_net
    P1_net.train_egwr(ds_vectors=train_frame,epochs=5,a_threshold=all_th[i],beta=0.5,l_rates=[0.1,0.08],context=3,regulated=0,train_mode=True)
    #test_p1_net
    P1_net.trained_nodes = P1_net.num_nodes
    data.train_test_set(ls_cls=trained_cls)
    te_frame = dataLoader(train=False,frames=True,mask_frame='aligned_masks')
    p2_w_test,p2_l_test = P1_net.test(ds_vectors=te_frame,test_accuracy=True,pool_size=9,ret_vecs=True)
    print(trained_cls,'p1_net_accuracy', P1_net.test_accuracy)










