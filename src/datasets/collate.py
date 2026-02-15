import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from dataset.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    batch = {}

    keys = dataset_items[0].keys()

    for key in keys:
        values = [item[key] for item in dataset_items]

        if isinstance(values[0], torch.Tensor):
            shapes = [v.shape for v in values]
            if all(s == shapes[0] for s in shapes):
                batch[key] = torch.stack(values, dim=0)
            else:
                max_len = max(v.shape[-1] for v in values)
                padded = []
                for v in values:
                    pad_size = max_len - v.shape[-1]
                    if pad_size > 0:
                        v = torch.nn.functional.pad(v, (0, pad_size))
                    padded.append(v)
                batch[key] = torch.stack(padded, dim=0)
        else:
            batch[key] = values

    return batch
