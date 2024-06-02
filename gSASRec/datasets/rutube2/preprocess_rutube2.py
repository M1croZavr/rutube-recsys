from collections import defaultdict
from pathlib import Path
import numpy as np
import json

DATASET_DIR = Path(__file__).parent
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
FILE_NAME = DATASET_DIR / "rutube_119258.txt"


# gSASRec paper used last interaction of each user for testing
# for 512 users, it also used the second last interaction for validation

def train_val_test_split():
    TRAIN_DIR.mkdir(exist_ok=True)
    VAL_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)

    rng = np.random.RandomState(123)
    user_items = defaultdict(list)
    with open(FILE_NAME) as f:
        for line in f:
            user_item = line.strip().split(" ")
            if len(user_item) != 2:
                continue
            user, item = user_item
            user = int(user) + 1
            item = int(item)
            user_items[user].append(item)
    # Filter users which have less than 5 items
    user_items = {user: items for user, items in user_items.items() if len(items) >= 5}
    items = {item for items in user_items.values() for item in items}
    # Renumber filtered items id to match 1, 2, 3, 4 ... pattern and items are enumerated from 1
    items_renumbering = dict(zip(sorted(list(items)), range(1, len(items) + 1)))
    user_items = {user: [items_renumbering[item] for item in items] for user, items in user_items.items()}
    num_interactions = sum(map(len, user_items.values()))
    dataset_stats = {
        "num_users": len(user_items),
        "num_items": len(items),
        "num_interactions": num_interactions
    }

    print("Dataset stats: ", json.dumps(dataset_stats, indent=4))
    with open(DATASET_DIR / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=4)

    train_sequences = []

    val_input_sequences = []
    val_gt_actions = []

    test_input_sequences = []
    test_gt_actions = []

    val_users = rng.choice(dataset_stats['num_users'], 512, replace=False)
    for user in user_items:
        if user in val_users:
            train_input_sequence = user_items[user][:-3]
            train_sequences.append(train_input_sequence)

            val_input_sequence = user_items[user][:-2]
            val_gt_action = user_items[user][-2]
            val_input_sequences.append(val_input_sequence)
            val_gt_actions.append(val_gt_action)

            test_input_sequence = user_items[user][:-1]
            test_gt_action = user_items[user][-1]
            test_input_sequences.append(test_input_sequence)
            test_gt_actions.append(test_gt_action)
        else:
            train_input_sequence = user_items[user][:-2]
            train_sequences.append(train_input_sequence)

            test_input_sequence = user_items[user][:-1]
            test_gt_action = user_items[user][-1]
            test_input_sequences.append(test_input_sequence)
            test_gt_actions.append(test_gt_action)

    with open(TRAIN_DIR / "input.txt", "w") as f:
        for sequence in train_sequences:
            f.write(" ".join([str(item) for item in sequence]) + "\n")

    with open(VAL_DIR / "input.txt", "w") as f:
        for sequence in val_input_sequences:
            f.write(" ".join([str(item) for item in sequence]) + "\n")

    with open(VAL_DIR / "output.txt", "w") as f:
        for action in val_gt_actions:
            f.write(str(action) + "\n")

    with open(TEST_DIR / "input.txt", "w") as f:
        for sequence in test_input_sequences:
            f.write(" ".join([str(item) for item in sequence]) + "\n")
    with open(TEST_DIR / "output.txt", "w") as f:
        for action in test_gt_actions:
            f.write(str(action) + "\n")


if __name__ == "__main__":
    train_val_test_split()
