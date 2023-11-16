from torch.utils.data import DataLoader
from hw_fs.collate_fn import Collate
# from type of args torch.utils.data


class LJLoader(DataLoader):
    def __init__(self, dataset, batch_size, batch_expand_size, num_workers):
        super().__init__(dataset=dataset, batch_size=batch_size * batch_expand_size,
                         shuffle=True, collate_fn=Collate, drop_last=True, num_workers=num_workers)
        self.batch_expand_size = batch_expand_size
