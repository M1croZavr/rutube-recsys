from aprec.evaluation.configs.bert4rec_repro_paper.common_benchmark_config import *
from aprec.evaluation.split_actions import LeaveOneOut

DATASET = "BERT4rec.rutube"
N_VAL_USERS = 1024
MAX_TEST_USERS = 2048
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
RECOMMENDERS = get_recommenders(
    filter_seen=True,
    filter_recommenders={"our_bert4rec_longer_seq", "b4vae_bert4rec", "recbole_bert4rec", "mf-bpr", "original_bert4rec"}
)
