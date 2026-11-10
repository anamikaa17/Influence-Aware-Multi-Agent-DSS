
import pandas as pd

TRAIN_PATH = "datasets/unsw_nb15/UNSW_NB15_training-set.csv"
TEST_PATH = "datasets/unsw_nb15/UNSW_NB15_testing-set.csv"


def main():

    print("LOADING UNSW-NB15 DATASET")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("\nDATASET SHAPES")
    print("Training:", train_df.shape)
    print("Testing:", test_df.shape)

    print("\nCOLUMN NAMES")
    print(train_df.columns.tolist())

    print("\nFIRST FIVE ROWS")
    print(train_df.head())

    print("\nTRAINING LABEL DISTRIBUTION")
    print(train_df["label"].value_counts())

    print("\nATTACK CATEGORY DISTRIBUTION")
    print(train_df["attack_cat"].value_counts(dropna=False))

    print("\nMISSING VALUES")
    print(train_df.isnull().sum().sort_values(ascending=False).head(15))
    print("\nTRAINING RECORDS:", len(train_df))
    print("TESTING RECORDS:", len(test_df))

    print("\nNORMAL vs ATTACK:")
    print(train_df["label"].value_counts())

    print("\nATTACK CATEGORIES:")
    print(train_df["attack_cat"].value_counts(dropna=False))

    print("\nTOTAL MISSING VALUES:")
    print(train_df.isnull().sum().sum())


if __name__ == "__main__":
    main()
