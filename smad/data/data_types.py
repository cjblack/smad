from torch.utils.data import Dataset

class SeqDataSet(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences  # list of tensors

    def __len__(self):
        return len(self.sequences) # sequence length

    def __getitem__(self, idx):
        return self.sequences[idx] # sequence index
