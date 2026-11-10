
import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from agents.feature_groups import AGENT_FEATURES


TRAIN_PATH = "datasets/unsw_nb15/UNSW_NB15_training-set.csv"
TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"
MODEL_DIR = "models"


class MLAgentTrainer:

    def __init__(self):
        self.models = {}
        os.makedirs(MODEL_DIR, exist_ok=True)

    # STEP 1: Prepare preprocessing and ML model
    def prepare_model(self, features, df):

        numerical_features = [
            col for col in features
            if pd.api.types.is_numeric_dtype(df[col])
        ]

        categorical_features = [
            col for col in features
            if not pd.api.types.is_numeric_dtype(df[col])
        ]

        print("Numerical features:", numerical_features)
        print("Categorical features:", categorical_features)

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ])

        classifier = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2
        )

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])

        return model

    # STEP 2: Train and evaluate all specialist agents
    def train_all(self):

        print("\nLOADING UNSW-NB15 DATA")

        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)

        predictions_by_agent = {}

        y_train = train_df["label"].astype(int)
        y_test = test_df["label"].astype(int)

        for agent_name, features in AGENT_FEATURES.items():

            print("\n" + "=" * 55)
            print("TRAINING:", agent_name)
            print("=" * 55)

            X_train = train_df[features].copy()
            X_test = test_df[features].copy()

            model = self.prepare_model(features, train_df)

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            self.models[agent_name] = model
            predictions_by_agent[agent_name] = predictions

            accuracy = accuracy_score(y_test, predictions)

            macro_f1 = f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )

            print("\nAccuracy:", round(accuracy, 4))
            print("Macro F1:", round(macro_f1, 4))

            print("\nClassification Report:")
            print(
                classification_report(
                    y_test,
                    predictions,
                    target_names=["Normal", "Attack"],
                    zero_division=0
                )
            )

            print("Confusion Matrix:")
            print(confusion_matrix(y_test, predictions))

            model_path = os.path.join(
                MODEL_DIR,
                agent_name.lower().replace(" ", "_") + ".joblib"
            )

            joblib.dump(model, model_path)

            print("\nModel saved:", model_path)

        return predictions_by_agent, y_test


if __name__ == "__main__":

    trainer = MLAgentTrainer()

    predictions, actual_labels = trainer.train_all()

    print("\nALL FOUR SPECIALIST AGENTS TRAINED SUCCESSFULLY")
