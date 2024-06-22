from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.evaluation.split_actions import LeaveOneOut, TestFileSplit
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import \
    FullMatrixTargetsBuilder
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
from aprec.evaluation.metrics.precision import Precision
from aprec.evaluation.metrics.recall import Recall
from aprec.recommenders.dnn_sequential_recommender.models.sasrec.sasrec import SASRec
from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender
from aprec.losses.bce import BCELoss
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.shifted_sequence_splitter import \
    ShiftedSequenceSplitter
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_per_positive_target import \
    NegativePerPositiveTargetBuilder

from tensorflow.keras.optimizers import Adam
from aprec.recommenders.metrics.ndcg import KerasNDCG


def original_ber4rec():
    recommender = VanillaBERT4Rec(max_seq_length=75, masked_lm_prob=0.3)
    return recommender


def dnn(model_arch, loss, sequence_splitter,
        val_sequence_splitter=SequenceContinuation,
        target_builder=FullMatrixTargetsBuilder,
        optimizer=Adam(),
        training_time_limit=3600, metric=KerasNDCG(40),
        max_epochs=10000
):
    return DNNSequentialRecommender(train_epochs=max_epochs, loss=loss,
                                    model_arch=model_arch,
                                    optimizer=optimizer,
                                    early_stop_epochs=100,
                                    batch_size=128,
                                    training_time_limit=training_time_limit,
                                    sequence_splitter=sequence_splitter,
                                    targets_builder=target_builder,
                                    val_sequence_splitter=val_sequence_splitter,
                                    metric=metric,
                                    debug=False
                                    )


vanilla_sasrec = lambda: dnn(
    SASRec(max_history_len=75,
           dropout_rate=0.5,
           num_heads=1,
           num_blocks=2,
           vanilla=True,
           embedding_size=64,
           ),
    BCELoss(),
    ShiftedSequenceSplitter,
    optimizer=Adam(beta_2=0.98),
    target_builder=lambda: NegativePerPositiveTargetBuilder(100),
    metric=BCELoss(),
)


recommenders = {
    "original_bert4rec": original_ber4rec,
    "original_sasrec": vanilla_sasrec
}


def get_recommenders(filter_seen: bool):
    result = {}
    for recommender_name in recommenders:
        if filter_seen:
            result[recommender_name] = \
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result


USERS_FRACTIONS = [1.0]
TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)
METRICS = [
    HIT(1), HIT(5), HIT(10),
    Precision(1), Precision(5), Precision(10),
    Recall(1), Recall(5), Recall(10),
    NDCG(1), NDCG(5), NDCG(10),
    MAP(1), MAP(5), MAP(10),
    MRR()
]
DATASET = "BERT4rec.rutube_week"
SPLIT_STRATEGY = TestFileSplit("data/bert4rec/rutube_week_test.txt")
RECOMMENDERS = get_recommenders(filter_seen=True)
