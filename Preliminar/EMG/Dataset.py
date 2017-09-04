import numpy as np
class Dataset:

    def __init__(self,data,labels):
        self._index_in_epoch = 0
        self._epochs_completed = 0
        self._data = data
        self._labels = labels
        self._num_examples = labels.shape[0]
        pass
#np.asarray([dataset.data[i] for i in idx])    
    
    @property
    def data(self):
        return self._data#, self._labels

    @property
    def labels(self):
        return self._labels
    
    def next_batch(self,batch_size,shuffle = True):
        start = self._index_in_epoch
        if start == 0 and self._epochs_completed == 0:
            idx = np.linspace(0, self._num_examples-1,self._num_examples)  # get all possible indexes
            idx=idx.astype(int)
            np.random.shuffle(idx)
            self._data = self.data[idx,:]  # get list of `num` random samples
            self._labels = self.labels[idx]  # get list of `num` random samples
    
        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            data_rest_part = self.data[start:self._num_examples,:]
            labels_rest_part = self.labels[start:self._num_examples]
            idx0 = np.arange(0, self._num_examples)  # get all possible indexes
            np.random.shuffle(idx0)  # shuffle indexes
            idx0=idx0.astype(int)
            self._data = self.data[idx0,:]  # get list of `num` random samples
            self._labels = self.labels[idx0]  # get list of `num` random samples
    
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples #avoid the case where the #sample != integar times of batch_size
            end =  self._index_in_epoch  
            data_new_part =  self._data[start:end,:]  
            labels_new_part =  self._labels[start:end]  
            return np.concatenate((data_rest_part, data_new_part)),np.concatenate((labels_rest_part, labels_new_part))
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end,:], self._labels[start:end]

if __name__ == "__main__":
    data=np.arange(0, 100).reshape(10, 10)
    labels=np.concatenate((np.arange(0, 5),np.arange(0, 5)))
    dataset = Dataset(data,labels)
    for i in range(4):
        print(dataset.next_batch(3))