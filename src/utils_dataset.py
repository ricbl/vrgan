from torch.utils.data import Dataset
import numpy as np

#dataset wrapper to load a dataset to memory for faster batch loading
class LoadToMemory(Dataset):
    def __init__(self, original_dataset):
        super().__init__()
        self.list_images = [original_dataset[0][0]]*len(original_dataset)
        self.list_labels = [original_dataset[0][1]]*len(original_dataset)
        indices_iterations = np.arange(len(original_dataset))
        
        for list_index, element_index in enumerate(indices_iterations): 
            self.list_images[list_index] = original_dataset[element_index][0]
            self.list_labels[list_index] = original_dataset[element_index][1]

    def __len__(self):
        return len(self.list_images)
    
    def __getitem__(self, index):
        return self.list_images[index], self.list_labels[index]

#class to iterate through two dataloaders at the same time. returns n_iter_per_epoch
# batches for each epoch. If the endd of one of the dataloaders is reached before reaching
# the end of the epoch, that dataloader is reset.
class IteratorLoaderDifferentSizes:
    def __init__(self, loader1, loader2, n_iter_per_epoch):
        self.loaders = [loader1, loader2]
        self.__iter__()
        self.count_iterations = 0
        self.n_iter_per_epoch = n_iter_per_epoch
        
    def __iter__(self):
        self.iterLoaders = [iter(loader) for loader in self.loaders]
        return self
    
    def nextI(self, this_iter):
        return next(this_iter,None)
    
    def __next__(self):
        if self.count_iterations >= self.n_iter_per_epoch:
            self.count_iterations = 0
            raise StopIteration
        current_batch_loader = []
        for i in range(len(self.loaders)):
            current_batch_loader.append(self.nextI(self.iterLoaders[i]))
            if current_batch_loader[i] is None:
                self.iterLoaders[i] = iter(self.loaders[i])
                current_batch_loader[i] = self.nextI(self.iterLoaders[i])
        self.count_iterations += 1
        return current_batch_loader
      
    next = __next__