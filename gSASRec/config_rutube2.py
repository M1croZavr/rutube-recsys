from ir_measures import nDCG, MAP, MRR, Precision, Recall

from gSASRec.config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='rutube2',
    sequence_length=200,
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=256,
    gbce_t=0.75,
    metrics=[nDCG @ 10, MAP @ 10, MRR, Precision @ 10, Recall @ 10],
    val_metric=nDCG @ 10
    # val_metric=[nDCG @ 10, MAP @ 10, MRR, Precision @ 10, Recall @ 10]
)


if __name__ == '__main__':
    print(MAP @ 10, MRR, Precision @ 10)
