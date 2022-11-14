from abc import ABC
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import pandas as pd
from pandas import DataFrame as DataFrame
from numpy import ndarray as ndarray
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from sklearn.metrics import plot_confusion_matrix
import joblib
import yaml
import matplotlib.pyplot as plt
from util.monitoring import calc_drift


class ModelInterface(ABC):

    def __init__(self, model_name: str):
        self.model_name = model_name

        self.pipeline = None
        self.hyper_params = None
        self.is_fitted = False

    def fit(self, pipeline: Pipeline, x_train: ndarray, y_train: ndarray, hyper_params: dict) -> None:
        self.pipeline = deepcopy(pipeline)

        self.pipeline.set_params(**hyper_params)
        self.pipeline.fit(x_train, y_train)

        self.hyper_params = hyper_params
        self.is_fitted = True

    def save(self) -> None:
        self._check_fitted()

        model_path = self._get_model_path()
        joblib.dump(self.pipeline, model_path)
        print(f"{self.model_name} (model) has been saved in {model_path}")

        hyper_params_path = self._get_model_hyper_params_path()
        with open(hyper_params_path, 'w') as file:
            yaml.dump(self.hyper_params, file, default_flow_style=False)
        print(f"{self.model_name} (hyper parameters) have been saved in {hyper_params_path}")

    def predict(self, input: ndarray) -> ndarray:
        self._check_fitted()

        preds = self.pipeline.predict(input)

        input["inference"] = preds
        inference_df = pd.DataFrame(data=input.values, columns=input.columns)
        inference_df.to_csv(self._get_model_inference_file(), index=False)

        return preds

    def test(self, x_test: ndarray, y_test: ndarray) -> None:
        self._check_fitted()

        fig = plot_confusion_matrix(self.pipeline, x_test, y_test)
        fig.figure_.suptitle("Confusion Matrix")
        plt.show()

    def load(self) -> None:
        self.pipeline = joblib.load(self._get_model_path())

        with open(self._get_model_hyper_params_path(), 'r') as file:
            self.hyper_params = yaml.load(file, Loader=yaml.SafeLoader)

        self.is_fitted = True

    def get_hyper_params(self):
        self._check_fitted()
        return self.hyper_params

    def get_inference_df(self) -> DataFrame:
        self._check_fitted()

        all_inferences = self._get_model_inference_dir().glob('*.csv')
        inference_df = pd.concat((pd.read_csv(f) for f in all_inferences), ignore_index=True)

        return inference_df

    def monitor_data_drift(self, x_train: ndarray, col_name: str) -> None:
        inference_df = self.get_inference_df()

        plt.hist(x_train[col_name], alpha=0.5, label='Ground Truth', histtype='step')
        plt.hist(inference_df[col_name], alpha=0.5, label='Observation', histtype='step')
        plt.legend()
        plt.title(f"Feature {col_name} Distribution of Ground Truth Data and Observation Data")
        plt.show()

    def calc_data_drift_score(self, x_train: ndarray, col_name: str) -> float:
        inference_df = self.get_inference_df()

        drift_score = calc_drift(gt_data=x_train, obs_data=inference_df, gt_col=col_name, obs_col=col_name)
        print(f"Drift score for {self.model_name} is: {drift_score}")

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError('Pipeline not fitted')

    def _get_model_root_dir(self) -> Path:
        model_root_dir = Path(self.model_name)
        model_root_dir.mkdir(parents=True, exist_ok=True)
        return model_root_dir

    def _get_model_inference_dir(self) -> Path:
        model_score_df_dir = self._get_model_root_dir() / 'inference'
        model_score_df_dir.mkdir(parents=True, exist_ok=True)
        return model_score_df_dir

    def _get_model_path(self):
        return self._get_model_root_dir() / f"{self.model_name}.pkl"

    def _get_model_hyper_params_path(self):
        return self._get_model_root_dir() / f"{self.model_name}.yaml"

    def _get_model_inference_file(self) -> Path:
        now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        return self._get_model_inference_dir() / f"{now_str}.csv"

    def __repr__(self) -> str:
        key_values = [f"{key}={value}" for key, value in vars(self).items()]
        return f"{self.__class__.__name__}: {', '.join(key_values)}"
