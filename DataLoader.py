import os
import numpy as np
import cv2
import scipy.misc
import pickle
import sys
import matplotlib.pyplot as plt
import PIL.Image as Image
import torch
import torchvision

class DataProcess():
    def __init__(self,data_path):
        self.data_path = data_path

    def frames_sample(self,sample_path,data_type,dic,file_name):   #sample_path: video_frames path      to be 10 frames per second  from 25 frames per second
        file_name = file_name[:-7]
        all_num = len(list(os.listdir(sample_path)))
        final_num = all_num // 2.5
        frame_id = np.round(np.linspace(1,all_num+1,final_num))
        for file in os.listdir(sample_path):
            if data_type == 'frames':
                frame_num = file.split('_')[1].split('.')[0]
                frame_num = int(frame_num)
                if frame_num not in frame_id:
                    os.remove(os.path.join(sample_path,file))
                # elif len(dic[file_name]) == 3:
                #     if frame_num<dic[file_name][0][0] or (frame_num>=dic[file_name][0][1] and frame_num<dic[file_name][1][0]) or (frame_num>=dic[file_name][1][1] and frame_num<dic[file_name][2][0]) or frame_num>=dic[file_name][2][1]:
                #         os.remove(os.path.join(sample_path,file))
                # elif len(dic[file_name]) == 4:
                #     if frame_num < dic[file_name][0][0] or (
                #             frame_num >= dic[file_name][0][1] and frame_num < dic[file_name][1][0]) or (
                #             frame_num >= dic[file_name][1][1] and frame_num < dic[file_name][2][0]) or (frame_num >= \
                #             dic[file_name][2][1] and frame_num < dic[file_name][3][0]) or frame_num > dic[file_name][3][1]:
                #         os.remove(os.path.join(sample_path, file))
            elif data_type == 'flows':
                frame_num = file.split('_')[2].split('.')[0]
                frame_num = int(frame_num)
                if frame_num not in frame_id:
                    os.remove(os.path.join(sample_path, file))
                # elif len(dic[file_name]) == 3:
                #     if frame_num<dic[file_name][0][0] or (frame_num>=dic[file_name][0][1] and frame_num<dic[file_name][1][0]) or (frame_num>=dic[file_name][1][1] and frame_num<dic[file_name][2][0]) or frame_num>=dic[file_name][2][1]:
                #         os.remove(os.path.join(sample_path,file))
                # elif len(dic[file_name]) == 4:
                #     if frame_num < dic[file_name][0][0] or (
                #             frame_num >= dic[file_name][0][1] and frame_num < dic[file_name][1][0]) or (
                #             frame_num >= dic[file_name][1][1] and frame_num < dic[file_name][2][0]) or (frame_num >= \
                #             dic[file_name][2][1] and frame_num < dic[file_name][3][0]) or frame_num > dic[file_name][3][1]:
                #         os.remove(os.path.join(sample_path, file))

    def frame_25(self):

        for cls in os.listdir(os.path.join(self.data_path,'frames')):
            for video in os.listdir(os.path.join(self.data_path,'frames',cls)):
                frames_list = list(os.listdir(os.path.join(self.data_path,'frames',cls,video)))
                frames_list.sort(key=lambda x: int(x[4:-4]))
                for num,frame in enumerate(frames_list):
                    if num > 79:
                        os.remove(os.path.join(self.data_path,'frames',cls,video,frame))

    def sample_all(self,sour_path):                                            #samplt for all frames and flow of every video
        with open('split_video.pkl','rb') as f:
            d = pickle.load(f)
        # for cls in os.listdir(os.path.join(sour_path,'frames')):
        #     for vide in os.listdir(os.path.join(sour_path,'frames',cls)):
        #         self.frames_sample(os.path.join(sour_path,'frames',cls,vide),data_type='frames',file_name=vide,dic=d)

        # for cls in os.listdir(os.path.join(sour_path,'flows')):
        #     for vide in os.listdir(os.path.join(sour_path,'flows',cls)):
        #         for channel in os.listdir(os.path.join(sour_path,'flows',cls,vide)):
        #             self.frames_sample(os.path.join(sour_path,'flows',cls,vide,channel),data_type='flows',file_name=vide,dic=d)

        for cls in os.listdir(os.path.join(sour_path, 'aligned_masks')):
            for vide in os.listdir(os.path.join(sour_path,'aligned_masks',cls)):
                self.frames_sample(os.path.join(sour_path,'aligned_masks',cls,vide),data_type='frames',file_name=vide,dic=d)

    def gray_resize(self,image_rgb):
        gray = cv2.cvtColor(image_rgb,cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray,(30,30))

        return gray

    def transform(self,mode):
        for root,dir,file in os.walk(top=os.path.join(self.data_path,mode)):
            if len(file)>=1 and file[0].split('.')[-1] == 'jpg':
                for img_name in file:
                    img = cv2.imread(os.path.join(root,img_name))
                    gray_img = self.gray_resize(img)

                    scipy.misc.imsave(os.path.join(root,img_name), gray_img)


    def train_test_set(self,ls_cls=['handclapping','jogging','running','boxing','handwaving','walking']):
        action_label = {'bend':0,'skip':1,'wave1':2,'jack':3,'jump':4,'pjump':5,'run':6,'side':7,'walk':8,'wave2':9}
        # action_label = {'handclapping':0,'jogging':1,'running':2,'boxing':3,'handwaving':4,'walking':5}
        dir_cls = list(action_label.keys())
        all_dic = {}
        train_dic = {}
        test_dic = {}

        train_name = ['person01','person02','person03','person04','person05']
        test_name = ['person25','person01','person23','person06','person21','person10','person19','person14','person17']
        # test_name = ['person25','person24']

        for cls in dir_cls:
            for file in os.listdir(os.path.join(self.data_path,'aligned_masks',cls)):
                all_dic[cls+'_'+file] = action_label[cls]

        for key,val in all_dic.items():
            name = key.split('_')[1]

            cls = key.split('_')[0]

            # for KTH
            # if cls in ls_cls:
            #     if name in test_name:
            #         test_dic[key] = val
            #     else:
            #         train_dic[key] = val


            #for weiamann
            if cls in ls_cls:
                if name == 'ido' or name == 'ira':
                    test_dic[key] = val
                else:
                    train_dic[key] = val


        out_file1 = open('train_dic_mask.pkl','wb')
        out_file2 = open('test_dic_mask.pkl','wb')
        pickle.dump(train_dic,out_file1)
        pickle.dump(test_dic,out_file2)
        out_file1.close()
        out_file2.close()

class dataLoader():

    def __init__(self,train=True,mask_frame='aligned_masks',frames=True,train_path ='./train_dic_mask.pkl',test_path='./test_dic_mask.pkl'):
        self.train = train
        self.frames = frames
        self.mask_frame = mask_frame
        self.train_path = train_path
        self.test_path = test_path
        self.begin = 0
        self.root = '/data1/ma_gps/Weizmann_Dataset/'

        if self.train == True:
            with open(self.train_path,'rb') as f:
                train_dic = pickle.load(f)

            self.sample_tuple = list(train_dic.items())
            self.end = len(self.sample_tuple)
        elif self.train == False:
            with open(self.test_path,'rb') as f:
                train_dic = pickle.load(f)
            self.sample_tuple = list(train_dic.items())
            self.end = len(self.sample_tuple)

    def __iter__(self):
        return self

    def __len__(self):
        return self.end


    def __next__(self):
            if self.begin < self.end:
                sample,label = self.sample_tuple[self.begin]
                vedio_samples = self.frames_patch(sample)
                self.begin += 1
                return (vedio_samples,label)
            else:
                self.begin = 0
                raise StopIteration

    def frames_patch(self,cls_name):
        if self.frames == True:
            vedio_patchs = []
            frams_list = os.listdir(os.path.join(self.root,self.mask_frame,cls_name.split('_')[0],cls_name.split('_',1)[1]))
            frams_list.sort(key=lambda x:int(x[4:-4]))

            for frame in frams_list:

                im = cv2.imread(os.path.join(self.root,self.mask_frame,cls_name.split('_')[0],cls_name.split('_',1)[1],frame))

                vedio_patchs.append(self.patchs(im))
            return vedio_patchs
        elif self.frames == False:
            vedio_x = []
            vedio_y = []
            for channel in os.listdir(os.path.join(self.root,'flows',cls_name.split('_')[0],cls_name.split('_',1)[1])):
                flow_list = os.listdir(os.path.join(self.root,'flows',cls_name.split('_')[0],cls_name.split('_',1)[1],channel))
                flow_list.sort(key=lambda x:int(x[7:-4]))
                for flow in flow_list:
                    im = cv2.imread(os.path.join(self.root,'flows',cls_name.split('_')[0],cls_name.split('_',1)[1],channel,flow),flags=-1)
                    if channel == 'flow_x':
                        vedio_x.append(self.patchs(im))
                    elif channel == 'flow_y':
                        vedio_y.append(self.patchs(im))
            vedio_flows = np.concatenate((vedio_x,vedio_y),axis=-1)
            return vedio_x                                       #first tyr to use the flow_x only


    def patchs(self,imag):
        img = Image.fromarray(np.uint8(imag))
        # print(type(img))
        img = img.resize((200, 200), Image.ANTIALIAS)
        mat = np.array(img, dtype=np.float32)
        mat = mat / 255.0

        mat = (mat - (0.5, 0.5, 0.5)) / (0.5, 0.5, 0.5)

        mat = np.swapaxes(mat, 1, 2)
        mat = np.swapaxes(mat, 0, 1)
        image = torch.from_numpy(mat)

        return image











if __name__ == '__main__':
    # kth = '/data1/ma_gps/KTH_dataset/'
    weizmann = '/data1/ma_gps/Weizmann_Dataset/'
    data = DataProcess(weizmann)
    # data.sample_all(data.data_path)
    # data.transform(mode='aligned_masks')
    data.train_test_set(ls_cls=['bend','skip','wave1','jack','jump','pjump','run','side','walk','wave2'])
    #data.frame_25()

    # dataset = dataLoader(train=True,frames=True,mask_frame='aligned_masks')
    # for data in dataset:
    #
    #     vedio,label = data
    #     print(label)
    #     for frame in vedio:
    #         print(frame)
    #         print(len(frame))
    #         sys.exit()


