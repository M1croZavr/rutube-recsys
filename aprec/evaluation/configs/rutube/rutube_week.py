from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.evaluation.split_actions import LeaveOneOut, TestFileSplit
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec

from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT

from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender


def original_ber4rec():
    recommender = VanillaBERT4Rec(max_seq_length=50, masked_lm_prob=0.5)
    return recommender


recommenders = {
    "original_bert4rec": lambda: original_ber4rec()
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
