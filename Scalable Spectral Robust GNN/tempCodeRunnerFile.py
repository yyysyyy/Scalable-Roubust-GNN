shuffle_idx = torch.randperm(len(dataset.x))
        x = x[shuffle_idx]