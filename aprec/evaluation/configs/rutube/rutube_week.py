from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.evaluation.split_actions import LeaveOneOut, TestFileSplit
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.recommenders.bert4recrepro.b4vae_bert4rec import B4rVaeBert4Rec
from aprec.recommenders.bert4recrepro.recbole_bert4rec import RecboleBERT4RecRecommender
from aprec.recommenders.dnn_sequential_recommender.dnn_sequential_recommender import DNNSequentialRecommender
from aprec.losses.mean_ypred_ploss import MeanPredLoss
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import \
    ItemsMaskingTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import \
    AddMaskHistoryVectorizer
from aprec.recommenders.dnn_sequential_recommender.models.bert4rec.bert4rec import BERT4Rec
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.last_item_splitter import SequenceContinuation
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import \
    FullMatrixTargetsBuilder
from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT
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
    recommender = VanillaBERT4Rec(max_seq_length=50, masked_lm_prob=0.5)
    return recommender


def b4rvae_bert4rec(epochs=None):
    # Default is 200 epochs
    recommender = B4rVaeBert4Rec(epochs=epochs)
    return recommender


def recbole_bert4rec(epochs=None):
    return RecboleBERT4RecRecommender(epochs=epochs)


def our_bert4rec(relative_position_encoding=False, sequence_len=50, rss=lambda n, k: 1, layers=2, arch=BERT4Rec,
                 masking_prob=0.2, max_predictions_per_seq=20):
    model = arch(max_history_len=sequence_len)
    recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=200,
                                           batch_size=64,
                                           training_time_limit=3600000,
                                           loss=MeanPredLoss(),
                                           debug=True, sequence_splitter=lambda: ItemsMasking(masking_prob=masking_prob,
                                                                                              max_predictions_per_seq=max_predictions_per_seq,
                                                                                              recency_importance=rss),
                                           targets_builder=lambda: ItemsMaskingTargetsBuilder(
                                               relative_positions_encoding=relative_position_encoding),
                                           val_sequence_splitter=lambda: ItemsMasking(force_last=True),
                                           metric=MeanPredLoss(),
                                           pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                           )
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
    SASRec(max_history_len=50,
           dropout_rate=0.2,
           num_heads=1,
           num_blocks=2,
           vanilla=True,
           embedding_size=50,
           ),
    BCELoss(),
    ShiftedSequenceSplitter,
    optimizer=Adam(beta_2=0.98),
    target_builder=lambda: NegativePerPositiveTargetBuilder(50),
    metric=BCELoss(),
)


recommenders = {
    # "original_bert4rec": original_ber4rec,
    # "our_bert4rec": our_bert4rec,  # XYETA
    "original_sasrec": vanilla_sasrec,
    # "recbole_bert4rec": recbole_bert4rec,
    # "b4vae_bert4rec": b4rvae_bert4rec,
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
METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), MAP(10)]
DATASET = "BERT4rec.rutube_week"
SPLIT_STRATEGY = TestFileSplit("data/bert4rec/rutube_week_test.txt")
RECOMMENDERS = get_recommenders(filter_seen=True)
