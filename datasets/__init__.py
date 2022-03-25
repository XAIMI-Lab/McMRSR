from . import cartesian_dataset


def get_datasets(opts):
    if opts.dataset == 'Clinical':
        trainset = cartesian_dataset.MRIDataset_Cartesian(opts, mode='TRAIN')
        valset = cartesian_dataset.MRIDataset_Cartesian(opts, mode='VALI')
        testset = cartesian_dataset.MRIDataset_Cartesian(opts, mode='TEST')

    else:
        raise NotImplementedError

    return trainset, valset, testset
