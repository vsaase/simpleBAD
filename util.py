import random

def create_train_idxs(n):
    shuffled = list(range(n))
    random.Random(0).shuffle(shuffled)
    train_ids = shuffled[:-len(shuffled)//9]
    val_ids = shuffled[len(train_ids):]
    return train_ids, val_ids

def create_patho_sample44():
    dfpatho = pd.read_csv("patho_all_t1_t1ce_flairfs_t2_t2star_adc_tracew_mprage.csv")
    shuffled = list(range(len(dfpatho)))
    random.Random(42).shuffle(shuffled)
    sample44 = shuffled[:44]
    return sample44