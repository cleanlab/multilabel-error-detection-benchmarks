"""
This script does the following:
- Configures the models based on the parameters in the config file.
- Saves the unitrained models and cross-validator to disk.
"""

import pickle
import yaml
import pathlib

import sklearn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def configure_classifier_dictionary(seed: int) -> dict[str, OneVsRestClassifier]:
    return {
        "log_reg": OneVsRestClassifier(
            LogisticRegression(random_state=seed), n_jobs=-1
        ),
        "rf": OneVsRestClassifier(RandomForestClassifier(random_state=seed), n_jobs=-1),
    }


def save_objects(
    train_params: dict,
    clf_dict: dict[str, OneVsRestClassifier],
    cross_validator: sklearn.model_selection.BaseCrossValidator,
) -> None:
    """Save the untrained classifier dictionary and the cross-validator to disk.

    The training stage should only depend on the training data, classifier dictionary,
    and cross-validator.
    """
    clf_dict_file = pathlib.Path(train_params["clf_dict"])
    cross_validator_file = pathlib.Path(train_params["cross_validator"])
    pickle.dump(clf_dict, open(clf_dict_file, "wb"))
    pickle.dump(cross_validator, open(cross_validator_file, "wb"))


if __name__ == "__main__":
    # Load params.yaml
    all_params = yaml.load(open("params.yaml"), Loader=yaml.FullLoader)
    train_params = all_params["train"]
    seed = train_params["seed"]
    cv_n_folds = train_params["cv_n_folds"]

    # Setup the classifiers and cross-validator
    clf_dict = configure_classifier_dictionary(seed=seed)
    cross_validator = sklearn.model_selection.StratifiedKFold(
        n_splits=cv_n_folds,
        shuffle=True,
        random_state=seed,
    )

    # Pickle the objects
    save_objects(
        train_params=train_params,
        clf_dict=clf_dict,
        cross_validator=cross_validator,
    )
