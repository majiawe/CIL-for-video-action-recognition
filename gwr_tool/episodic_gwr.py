"""
gwr-tb :: Episodic-GWR
@last-modified: 25 January 2019
@author: German I. Parisi (german.parisi@gmail.com)

"""

import numpy as np
import math
from gwr_tool.gammagwr import GammaGWR
import sys
from tqdm import tqdm
import logging
from heapq import nlargest
import torch
import torchvision
from feature_extract import resnet18

logger = logging.getLogger('log')
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler('/home/ma/CIL-for-video-action-recognition/log/log.txt')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)

logger.addHandler(handler)
# logger.addHandler(console)

devices = [5]
feature_extractor = resnet18(pretrained=True)
feature_extractor.cuda(device=devices[0])
feature_extractor.eval()

all_img = []
all_img_test = []

def img_vector(img):
    fra = img.cuda(device=devices[0])
    fra = torch.unsqueeze(fra, dim=0).float()
    _, feature_vector = feature_extractor(fra)
    feature_vector = torch.squeeze(feature_vector, dim=0).detach().cpu().numpy()
    return feature_vector







class EpisodicGWR(GammaGWR):

    def __init__(self):
        self.iterations = 0
    
    def init_network(self, ds,dimension, e_labels, num_context) -> None:
        
        assert self.iterations < 1, "Can't initialize a trained network"
        assert ds is not None, "Need a dataset to initialize a network"
        
        # Lock to prevent training
        self.locked = False

        # Start with 2 neurons
        self.num_nodes = 2
        self.trained_nodes = None
        self.dimension = dimension
        # self.patch_num = patch_num
        self.num_context = num_context
        self.depth = self.num_context + 1
        empty_neuron = np.zeros((self.depth, self.dimension))
        self.weights = [empty_neuron, empty_neuron]
        
        # Global context
        self.g_context = np.zeros((self.depth, self.dimension))
        
        # Create habituation counters
        self.habn = [1, 1]
        
        # Create edge and age matrices
        self.edges = np.ones((self.num_nodes, self.num_nodes))
        self.ages = np.zeros((self.num_nodes, self.num_nodes))
        
        # Temporal connections
        self.temporal = np.zeros((self.num_nodes, self.num_nodes))

        # Label histogram
        self.num_labels = e_labels

        self.alabels = []

        self.train_one = 0

        for l in range(0, len(self.num_labels)):
            self.alabels.append(-np.ones((self.num_nodes, self.num_labels[l])))
        init_ind = list(range(0, self.num_nodes))
        for i in range(0, len(init_ind)):
            for vedios in ds:
                fra = vedios[0][i]

                # input_img = merge(fra, train=True,init=True)
                # frame = img_vector(input_img)

                feature_vector = img_vector(fra)

                self.weights[i][0] = feature_vector
                break
        # Context coefficients
        self.alphas = self.compute_alphas(self.depth)
            
    def update_temporal(self, current_ind, previous_ind, **kwargs) -> None:
        new_node = kwargs.get('new_node', False)
        if new_node:
            self.temporal = super().expand_matrix(self.temporal)
        if previous_ind != -1 and previous_ind != current_ind:
            self.temporal[previous_ind, current_ind] += 1

    def update_labels(self, bmu, label, **kwargs) -> None:
        new_node = kwargs.get('new_node', False)        
        if not new_node:
            for l in range(0, len(self.num_labels)):
                for a in range(0, self.num_labels[l]):
                    if a == label[l]:
                        self.alabels[l][bmu, a] += self.a_inc
                    else:
                        if label[l] != -1:
                            self.alabels[l][bmu, a] -= self.a_dec
                            if (self.alabels[l][bmu, a] < 0):
                                self.alabels[l][bmu, a] = 0              
        else:
            for l in range(0, len(self.num_labels)):
                new_alabel = np.zeros((1, self.num_labels[l]))
                if label[l] != -1:
                    new_alabel[0, int(label[l])] = self.a_inc
                self.alabels[l] = np.concatenate((self.alabels[l], new_alabel), axis=0)

    def remove_isolated_nodes(self) -> None:
        if self.trained_nodes == None:
            if self.num_nodes > 2:
                ind_c = 0
                rem_c = 0
                while (ind_c < self.num_nodes):
                    neighbours = np.nonzero(self.edges[ind_c])
                    if len(neighbours[0]) < 1:
                        if self.num_nodes > 2:
                            self.weights.pop(ind_c)
                            self.habn.pop(ind_c)
                            for d in range(0, len(self.num_labels)):
                                d_labels = self.alabels[d]
                                self.alabels[d] = np.delete(d_labels, ind_c, axis=0)
                            self.edges = np.delete(self.edges, ind_c, axis=0)
                            self.edges = np.delete(self.edges, ind_c, axis=1)
                            self.ages = np.delete(self.ages, ind_c, axis=0)
                            self.ages = np.delete(self.ages, ind_c, axis=1)
                            #self.temporal = np.delete(self.temporal, ind_c, axis=0)
                            #self.temporal = np.delete(self.temporal, ind_c, axis=1)
                            self.num_nodes -= 1
                            rem_c += 1
                        else: return
                    else:
                        ind_c += 1
        else:
            if self.num_nodes > 2:
                ind_c = 0
                rem_c = 0
                while (ind_c < self.num_nodes):
                    if ind_c > self.trained_nodes:
                        neighbours = np.nonzero(self.edges[ind_c])
                        if len(neighbours[0]) < 1:
                            if self.num_nodes > 2:
                                self.weights.pop(ind_c)
                                self.habn.pop(ind_c)
                                for d in range(0, len(self.num_labels)):
                                    d_labels = self.alabels[d]
                                    self.alabels[d] = np.delete(d_labels, ind_c, axis=0)
                                self.edges = np.delete(self.edges, ind_c, axis=0)
                                self.edges = np.delete(self.edges, ind_c, axis=1)
                                self.ages = np.delete(self.ages, ind_c, axis=0)
                                self.ages = np.delete(self.ages, ind_c, axis=1)
                                #self.temporal = np.delete(self.temporal, ind_c, axis=0)
                                #self.temporal = np.delete(self.temporal, ind_c, axis=1)
                                self.num_nodes -= 1
                                rem_c += 1
                            else: return
                        else:
                            ind_c += 1
                    else:
                        ind_c += 1
        print ("(-- Removed %s neuron(s))" % rem_c)

    def remove_isolated_nodes_original(self) -> None:
        if self.num_nodes > 2:
            ind_c = 0
            rem_c = 0
            while (ind_c < self.num_nodes):
                neighbours = np.nonzero(self.edges[ind_c])            
                if len(neighbours[0]) < 1:
                    if self.num_nodes > 2:
                        self.weights.pop(ind_c)
                        self.habn.pop(ind_c)
                        for d in range(0, len(self.num_labels)):
                            d_labels = self.alabels[d]
                            self.alabels[d] = np.delete(d_labels, ind_c, axis=0)
                        self.edges = np.delete(self.edges, ind_c, axis=0)
                        self.edges = np.delete(self.edges, ind_c, axis=1)
                        self.ages = np.delete(self.ages, ind_c, axis=0)
                        self.ages = np.delete(self.ages, ind_c, axis=1)
                        #self.temporal = np.delete(self.temporal, ind_c, axis=0)
                        #self.temporal = np.delete(self.temporal, ind_c, axis=1)
                        self.num_nodes -= 1
                        rem_c += 1
                    else: return
                else:
                    ind_c += 1
            print ("(-- Removed %s neuron(s))" % rem_c)
         
    def train_egwr(self, ds_vectors, epochs, a_threshold, beta,
                   l_rates, context, regulated,train_mode=False) -> None:
        
        assert not self.locked, "Network is locked. Unlock to train."


        self.samples = 2000
        self.max_epochs = epochs
        self.a_threshold = a_threshold   
        self.epsilon_b, self.epsilon_n = l_rates
        self.beta = beta
        self.regulated = regulated
        self.context = context
        if not self.context:
            self.g_context.fill(0)
        self.hab_threshold = 0.1
        self.tau_b = 0.3
        self.tau_n = 0.1
        self.max_nodes = self.samples # OK for batch, bad for incremental
        self.max_neighbors = 6
        self.max_age = 70     #39000
        self.new_node = 0.5
        self.a_inc = 1
        self.a_dec = 0.1
        self.mod_rate = 0.8

            
        # Start training
        error_counter = np.zeros(self.max_epochs)

        previous_ind = -1
        for epoch in range(0, self.max_epochs):
            for i1,vedio in tqdm(enumerate(ds_vectors)):
                frames,lab = vedio
                previous_bmu = np.zeros(( self.depth, self.dimension))
                previous_context = np.zeros((self.depth,self.dimension))
                for i2,frame in enumerate(frames):
                    # Generate input sample

                    frame = img_vector(frame)
                    self.g_context[0] = frame
                    label =[lab]
                    # Update global context
                    #self.g_context[1] = np.array(frame) - np.array(pre_frame)

                    for z in range(1, self.depth):
                        # self.g_context[z] = (self.beta * previous_bmu[z]) + ((1-self.beta) * previous_bmu[z-1])
                        self.g_context[z] = previous_context[z-1]
                        #self.g_context[z] = previous_bmu[z-1]

                    previous_context = self.g_context
                    # Find the best and second-best matching neurons
                    b_index, b_distance, s_index = super().find_bmus(self.g_context, s_best = True,)
                    b_label = np.argmax(self.alabels[0][b_index])
                    misclassified = b_label != label[0]
                    # Quantization error
                    error_counter[epoch] += b_distance



                    # Compute network activity
                    a = math.exp(-b_distance)


                    # if a > self.a_threshold:
                    #     self.flag[b_index] = 1
                    #     b0_rate = self.habn[b_index] * b_rate
                    #
                    #     if b0_rate <= 0.08:
                    #         b0_rate = 0.08
                    # else:
                    #     if self.flag[b_index] == 1:
                    #         self.habn[b_index]**2 * b_rate
                    #     b0_rate = b_rate

                    # Store BMU at time t for t+1
                    previous_bmu = self.weights[b_index]


                    if (not self.regulated) or (self.regulated and misclassified):

                        if (a < self.a_threshold
                            and self.habn[b_index] < self.hab_threshold
                            and self.num_nodes < self.max_nodes):
                            # Add new neuron
                            n_index = self.num_nodes
                            super().add_node(b_index)


                            # Add label histogram
                            self.update_labels(n_index, label, new_node = True)

                            # Update edges and ages
                            super().update_edges(b_index, s_index, new_index = n_index)

                            # Update temporal connections
                            #self.update_temporal(n_index, previous_ind, new_node = True)

                            # Habituation counter
                            super().habituate_node(n_index, self.tau_b, new_node = True)

                        else:
                            # Habituate BMU

                            super().habituate_node(b_index, self.tau_b)

                            # Update BMU's weight vector

                            if self.regulated and misclassified:
                                b_rate *= self.mod_rate
                                n_rate *= self.mod_rate
                            else:
                                # Update BMU's label histogram
                                self.update_labels(b_index, label)

                            # n0_rate = self.habn[s_index] * n_rate
                            b_rate, n_rate = self.epsilon_b, self.epsilon_n

                            # b0_rate = b_rate * self.habn[b_index]
                            # n0_rate = n_rate * self.habn[s_index]
                            #
                            # if b0_rate < 0.1:
                            #     b0_rate = 0.1
                            # if n0_rate < 0.01:
                            #     n0_rate = 0.01
                            # if n0_rate <= 0.001:
                            #     n0_rate = 0.001
                            super().update_weight(b_index, b_rate)

                            # Update BMU's edges // Remove BMU's oldest ones
                            super().update_edges(b_index, s_index)

                            # Update temporal connections
                            #self.update_temporal(b_index, previous_ind)

                            # Update BMU's neighbors
                            super().update_neighbors(b_index, n_rate)
                        
                self.iterations += 1
                    
                previous_ind = b_index

            # Remove old edges
            super().remove_old_edges()

            # Average quantization error (AQE)
            error_counter[epoch] /= self.samples
            
            print ("(Epoch: %s, NN: %s, ATQE: %s)" % (epoch + 1, self.num_nodes, error_counter[epoch]))
            
        # Remove isolated neurons
            if epoch != self.max_epochs - 1:
                self.remove_isolated_nodes()
        #     else:
        #         pass
        # if train_mode == True:
        #     self.memory_consolidate(ds_vectors)

    # def memory_consolidate(self,train_data):
    #     self.consoli_nodes = np.zeros(self.num_nodes)
    #     input = np.zeros((self.depth,self.dimension))
    #     for video in train_data:
    #         pre_input = np.zeros((self.depth,self.dimension))
    #
    #         frames,_ = video
    #         for frame in frames:
    #             input[0] = frame
    #             for z in range(1,self.depth):
    #                 input[z] = pre_input[z-1]
    #             pre_input = input
    #             b_index, b_distance = super().find_bmus(input)
    #             self.consoli_nodes[b_index] = 1                         #find the best match node and mark it
    #     stable_node = np.nonzero(self.consoli_nodes)
    #     for ind in stable_node[0]:
    #         self.habn[ind] *= 0.1
    #     self.train_one = 1
        


    def pool(self,vector,pool_size):
        new_vct = []
        length = len(vector) // pool_size
        for i in range(length):
            new_vct.append(max(vector[i * pool_size:(i + 1) * pool_size]))
        return new_vct




    def test(self, ds_vectors,pool_size=9, **kwargs):               #could be pooling layer
        test_accuracy = kwargs.get('test_accuracy', False)
        test_vecs = kwargs.get('ret_vecs', False)
        test_samples = len(ds_vectors)
        self.bmus_index = []
        self.bmus_weight = []
        self.bmus_label = []
        self.bmus_activation = []
        self.weight_label = []
        self.a = []

        
        if test_accuracy:
            acc_counter = np.zeros(len(self.num_labels))
        
        for ind,vedio in enumerate(ds_vectors):
            self.bmus_index.append([])
            self.bmus_weight.append([])
            self.bmus_activation.append([])
            self.bmus_label.append([])
            # self.a.append([])

            frames,label = vedio
            self.weight_label.append(label)
            input_context = np.zeros((self.depth, self.dimension))
            pre_bmu_weight = np.zeros((self.depth,self.dimension))
            pre_context = np.zeros((self.depth,self.dimension))
            # pre_frame = np.zeros(self.dimension)
            for frame in frames:
                # Find the BMU
                frame = img_vector(frame)
                input_context[0] = frame
                #input_context[1] = np.array(frame) - np.array(pre_frame)
                for z in range(1, self.depth):
                    input_context[z] = pre_context[z-1]
                pre_context = input_context
                #     input_context[z] = input_context[z - 1]
                     # input_context[z] = (self.beta * pre_bmu_weight[z]) + ((1-self.beta) * pre_bmu_weight[z-1])
                b_index, b_distance = super().find_bmus(input_context,train=False)
                # self.a[ind].append((math.exp(-b_distance),b_index))
                self.bmus_index[ind].append(b_index)

                if test_vecs == True:
                    self.bmus_weight[ind].append(self.pool(self.weights[b_index][0],pool_size=pool_size))
                # self.bmus_activation[ind].append(math.exp(-b_distance))
            #for l in range(0, len(self.num_labels)):



                pre_bmu_weight = self.weights[b_index]                                    #has been modified by maJw
                pre_frame = frame
                # for j in range(1, self.depth):
                #     input_context[j] = input_context[j-1]

            if test_accuracy:
                # bb_ind = nlargest(6, self.a[ind])
                # s_ind = list(set([l[1] for l in bb_ind]))
                for i in self.bmus_index[ind]:

                    self.bmus_label[ind].append(np.argmax(self.alabels[0][i]))

                #for l in range(0, len(self.num_labels)):
                pred_label = np.bincount(self.bmus_label[ind])
                pred_label = np.argmax(pred_label)

                if pred_label == label:
                    acc_counter += 1
                print('pred_label: ',pred_label,'****label: ',label)
        if test_accuracy: self.test_accuracy =  acc_counter / test_samples
            
        if test_vecs:
            # s_labels = -np.ones((1, test_samples))
            # s_labels[0] = self.bmus_label[1]
            return self.bmus_weight, self.weight_label
