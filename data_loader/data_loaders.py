from base import BaseDataLoader
import data_loader.mydataloader_tahoe as tahoe
import data_loader.mydataloader_synth_ex1 as synth_ex1
import data_loader.mydataloader_synth_ex2 as synth_ex2

from data_loader.seq_util import seq_collate_fn






class TahoeDataLoader(BaseDataLoader):
    def __init__(self,
                 batch_size,
                 data_dir='DATA',
                 split='train',
                 shuffle=True,
                 collate_fn=seq_collate_fn,
                 num_workers=1):

        assert split in ['train', 'valid', 'test']
        self.dataset = tahoe.TahoeDataset(data_dir, split)
        self.data_dir = data_dir
        self.split = split
        
        super().__init__(self.dataset,
                         batch_size,
                         shuffle,
                         0.0,
                         num_workers,
                         seq_collate_fn)





class SynthEx1DataLoader(BaseDataLoader):
    def __init__(self,
                 batch_size,
                 data_dir='DATA',
                 split='train',
                 shuffle=True,
                 collate_fn=seq_collate_fn,
                 num_workers=1):

        assert split in ['train', 'valid', 'test']
        self.dataset = synth_ex1.SynthEx1Dataset(data_dir, split)
        self.data_dir = data_dir
        self.split = split
        
        super().__init__(self.dataset,
                         batch_size,
                         shuffle,
                         0.0,
                         num_workers,
                         seq_collate_fn)
        
        
        

class SynthEx2DataLoader(BaseDataLoader):
    def __init__(self,
                 batch_size,
                 data_dir='DATA',
                 split='train',
                 shuffle=True,
                 collate_fn=seq_collate_fn,
                 num_workers=1):

        assert split in ['train', 'valid', 'test']
        self.dataset = synth_ex2.SynthEx2Dataset(data_dir, split)
        self.data_dir = data_dir
        self.split = split
        
        super().__init__(self.dataset,
                         batch_size,
                         shuffle,
                         0.0,
                         num_workers,
                         seq_collate_fn)
        
        



