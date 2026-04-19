import pandas as pd 

df = pd.read_csv("dataset_867_visualizing_livestock.csv")

def check_imbalance(df, target_column):
    class_distribution = df[target_column].value_counts(normalize = True)

    print("\nClass distribution (proportions):")
    print(class_distribution)

    print("\nClass distribution (%): ")
    print(class_distribution * 100)

    return class_distribution 

print(df)
check_imbalance(df, "binaryClass")