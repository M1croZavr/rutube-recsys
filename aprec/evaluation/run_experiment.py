import os
import sys
import importlib.util
import json
import mmh3

from aprec.utils.os_utils import shell
from aprec.evaluation.evaluate_recommender import RecommendersEvaluator
from aprec.datasets.datasets_register import DatasetsRegister

import tensorflow as tf


def config():
    """ from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path"""

    # e.g. configs/rutube/rutube_benchmark1.py
    spec = importlib.util.spec_from_file_location("config", sys.argv[1])

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    if len(sys.argv) > 2:
        # e.g. ./results/rutube_benchmark_2024_01_01T12_12_12/experiment_1.json
        config.out_file = open(sys.argv[2], 'w')
        # e.g. ./results/rutube_benchmark_2024_01_01T12_12_12/
        config.out_dir = os.path.dirname(sys.argv[2])
    else:
        config.out_file = sys.stdout
        config.out_dir = os.getcwd()

    return config


def real_hash(obj):
    str_val = str(obj)
    result = (mmh3.hash(str_val) + (1 << 31)) * 1.0 / ((1 << 32) - 1)
    return result


def run_experiment(config):
    result = []
    # e.g. "BERT4rec.rutube"
    print(f"Dataset: {config.DATASET}")
    print("Reading  data...")
    # List of all actions (Action instances) from .txt file
    all_actions = [action for action in DatasetsRegister()[config.DATASET]()]
    print("DONE\n")
    callbacks = ()
    if hasattr(config, 'CALLBACKS'):
        callbacks = config.CALLBACKS

    for users_fraction in config.USERS_FRACTIONS:
        # Фильтрация количества user_id, которые пойдут в обучение и валидацию
        # Если 1, то берем всех из датасета
        # user_fraction - доля пользователей из общей выборки, которых мы берем в actions
        every_user = 1 / users_fraction
        print("Use one out of every {} users ({}% fraction)".format(every_user, users_fraction * 100))
        actions = list(
            filter(
                lambda action: real_hash(action.user_id) < users_fraction,
                all_actions
            )
        )
        print("Number of actions in dataset: {}".format(len(actions)))
        item_id_set = set([action.item_id for action in actions])
        user_id_set = set([action.user_id for action in actions])

        if hasattr(config, 'N_VAL_USERS'):
            n_val_users = config.N_VAL_USERS
        else:
            n_val_users = len(user_id_set) // 10

        if hasattr(config, 'USERS'):
            users = config.USERS()
        else:
            users = None

        if hasattr(config, 'ITEMS'):
            items = config.ITEMS()
        else:
            items = None

        if hasattr(config, 'RECOMMENDATIONS_LIMIT'):
            recommendations_limit = config.RECOMMENDATIONS_LIMIT
        else:
            recommendations_limit = 900

        print("Number of items in the dataset: {}".format(len(item_id_set)))
        print("Number of users in the dataset: {}".format(len(user_id_set)))
        print("Number of val_users: {}\n".format(n_val_users))
        print("Evaluating...")

        # LeaveOneOut(MAX_TEST_USERS)
        # Его __call__ принимает actions и возвращает train и test списки из actions (отсортированные по timestamp),
        # где в train actions до последнего timestamp, а в test action последнего timestamp для этого user_id
        # Количество user_id в test определяется MAX_TEST_USERS и выбираются случайно

        # WeekOut
        # Его __call__ принимает actions и возвращает train и test, где тест просто берется из другого файла
        data_splitter = config.SPLIT_STRATEGY
        target_items_sampler = None
        if hasattr(config, "TARGET_ITEMS_SAMPLER"):
            # PopTargetItemsSampler(target_size)
            # Дополняет test случайными (пропорционально частотной вероятности) айтемами до target_size
            target_items_sampler = config.TARGET_ITEMS_SAMPLER

        filter_cold_start = True

        if hasattr(config, "FILTER_COLD_START"):
            filter_cold_start = config.FILTER_COLD_START

        recommender_evaluator = RecommendersEvaluator(
            # All actions from .txt
            actions,
            # Dict of recommender_name -> function which returns a recommender
            config.RECOMMENDERS,
            # Metrics functions list
            config.METRICS,
            # e.g. ./results/rutube_benchmark_2024_01_01T12_12_12/
            config.out_dir,
            # Train test splitter
            data_splitter,
            # FOR WHAT?
            n_val_users,
            recommendations_limit,
            callbacks,
            users=users,
            items=items,
            experiment_config=config,
            # Sampling negatives for test items
            target_items_sampler=target_items_sampler,
            # Remove those test actions which users were not presented in a train sample
            remove_cold_start=filter_cold_start
        )

        if hasattr(config, 'FEATURES_FROM_TEST'):
            recommender_evaluator.set_features_from_test(config.FEATURES_FROM_TEST)
        result_for_fraction = recommender_evaluator()
        result_for_fraction['users_fraction'] = users_fraction
        result_for_fraction['num_items'] = len(item_id_set)
        result_for_fraction['num_users'] = len(user_id_set)
        result.append(result_for_fraction)
        write_result(config, result)
        shell(f"python3 statistical_signifficance_test.py --predictions-dir={config.out_dir}/predictions/ "
              f"--output-file={config.out_dir}/statistical_signifficance.json")


def get_max_test_users(config):
    if hasattr(config, 'MAX_TEST_USERS'):
        max_test_users = config.MAX_TEST_USERS
    else:
        max_test_users = 943  # number of users in movielens 100k dataset
    return max_test_users


def write_result(config, result):
    if config.out_file != sys.stdout:
        config.out_file.seek(0)
    config.out_file.write(json.dumps(result, indent=4))
    if config.out_file != sys.stdout:
        config.out_file.truncate()
        config.out_file.flush()


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    config = config()
    run_experiment(config)
