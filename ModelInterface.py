from abc import ABC
from pathlib import Path
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import pandas as pd
from joblib import dump, load
from monitoring import calc_drift


class ModelInterface(ABC):

    def __init__(self, model_name: str, *args, **kwargs):
        self.model_name = model_name

        self.pipeline = None
        self.is_fitted = False
        self.pipeline_analytics = None

    def fit(self, pipeline: Pipeline, train_x, train_y, hyper_params):
        self.pipeline = deepcopy(pipeline)

        self.pipeline.set_params(**hyper_params)
        self.pipeline.fit(train_x, train_y)
        self.is_fitted = True

    def save(self):
        dump(self.pipeline, self._get_model_root_dir() / f"{self.model_name}.fit")

    def load(self):
        self.pipeline = load(self._get_model_root_dir() / f"{self.model_name}.fit")
        self.is_fitted = True

    def predict(self, x_score):
        self.check_fitted()
        y_score = self.pipeline.predict(x_score)

        x_score["inference"] = y_score
        score_df = pd.DataFrame(data=x_score.values, columns=x_score.columns)
        score_df.to_csv(self._get_model_inference_dir())

        return y_score

    @staticmethod
    def monitor_data_drift(model_name: str, x_train, x_score, col_name):
        calc_drift(gt_data=x_train, obs_data=x_score, gt_col=col_name, obs_col=col_name)

    def _check_fitted(self):
        if not self.is_fitted:
            raise NotFittedError('Pipeline not fitted')

    def _get_model_root_dir(self):
        model_root_dir = Path(self.model_name)
        model_root_dir.mkdir(parents=True, exist_ok=True)
        return model_root_dir

    def _get_model_inference_dir(self):
        model_score_df_dir = self._get_model_root_dir() / 'inference'
        model_score_df_dir.mkdir(parents=True, exist_ok=True)
        return model_score_df_dir

    def __repr__(self):
        key_values = [f"{key}={value}" for key, value in vars(self).items()]
        return f"{self.__class__.__name__}: {', '.join(key_values)}"
