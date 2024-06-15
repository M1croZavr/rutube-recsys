from ir_measures import nDCG, MAP, MRR, Precision, Recall

from gSASRec.config import GSASRecExperimentConfig


config = GSASRecExperimentConfig(
    dataset_name='rutube_week',
    sequence_length=50,
    # embedding_dim=50,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.5,
    # negs_per_pos=100,
    gbce_t=0.75,
    metrics=[
        Precision @ 1, Precision @ 5, Precision @ 10,
        Recall @ 1, Recall @ 5, Recall @ 10,
        nDCG @ 5, nDCG @ 10,
        MRR, MAP @ 10
    ],
    val_metric=nDCG @ 10
)
