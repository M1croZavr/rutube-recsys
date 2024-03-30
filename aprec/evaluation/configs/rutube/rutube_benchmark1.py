from aprec.evaluation.samplers.pop_sampler import PopTargetItemsSampler
from aprec.evaluation.split_actions import LeaveOneOut
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec

from aprec.evaluation.metrics.ndcg import NDCG
from aprec.evaluation.metrics.mrr import MRR
from aprec.evaluation.metrics.map import MAP
from aprec.evaluation.metrics.hit import HIT

from aprec.recommenders.filter_seen_recommender import FilterSeenRecommender

# Use every fraction * 100% user of all actions
USERS_FRACTIONS = [1.0]  # + [0.1, 0.05]


def original_ber4rec(training_steps):
    recommender = VanillaBERT4Rec(num_train_steps=training_steps)
    return recommender


recommenders = {
    "original_bert4rec-25000": lambda: original_ber4rec(25000)
}

# Параметр - до скольки айтемов дополнять тестовый список actionов
# Sampling - оценка по сэмплингу, то есть ранжирование по тестовым айтемам, которые случайно засемплены
TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(10)
METRICS = [HIT(1), HIT(5), HIT(10), MAP(1), MAP(5), MAP(10), NDCG(5), NDCG(10), MRR()]


def get_recommenders(filter_seen: bool, filter_recommenders=None):
    if filter_recommenders is None:
        filter_recommenders = set()
    result = {}
    for recommender_name in recommenders:
        if recommender_name in filter_recommenders:
            continue

        if filter_seen:
            result[recommender_name] = \
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]

    return result


DATASET = "BERT4rec.rutube"
N_VAL_USERS = 1024
# MAX_TEST_USERS - в случае LeaveOneOut это сколько пользователей будут оцениваться по последнему действию
MAX_TEST_USERS = 2048
# Returns lists of sorted actions with the last timestamp user's item for the test list
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(filter_seen=True)
