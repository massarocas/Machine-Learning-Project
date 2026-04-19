import logging

try:
    from sklearn.model_selection import train_test_split 
except ImportError:
    from sklearn.cross_validation import train_test_split 
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression 

from mla.linear_models import LinearRegression, LogisticRegression 
from mla.metrics.metrics import mean_squared_error, accuracy 

# change to DEBUG to see convergence 
logging.basicConfig(level=logging.ERROR)

def regression():
    # generate a randam regression problem 
    x, y = make_regression(
        n_samples = 10000,
        n_features = 100, 
        n_informative = 75,
        n_targets = 1,
        noise = 0.05,
        random_state = 1111,
        bias = 0.5,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.25, random_state = 1111
    )

    model = LinearRegression(lr=0.01, max_iters=2000, penalty="l2", C=0.03)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("regression mse", mean_squared_error(y_test, predictions))



def confusion_matrix_manual(y_true, y_pred):
    tp = fp = tn = fn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1

    return tp, fp, tn, fn 

def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) != 0 else 0 

def recall (tp, fn):
    return tp / (tp + fn) if (tp + fn) != 0 else 0 

def f1(p, r):
    return 2*p*r / (p+r) if (p + r) != 0 else 0 

def balanced_accuracy(tp, fp, tn, fn):
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0 
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0 
    return (tpr + tnr) / 2


def classification():
    X, y = make_classification(
        n_samples=5000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        n_classes=2,
        weights=[0.95, 0.05],  # imbalance
        random_state=1111
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1111
    )

    model = LogisticRegression(lr=0.01, max_iters=500, penalty="l1", C=0.01)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    tp, fp, tn, fn = confusion_matrix_manual(y_test, predictions)

    acc = accuracy(y_test, predictions)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1_score = f1(prec, rec)
    bal_acc = balanced_accuracy(tp, fp, tn, fn)

    print("\nClassification Metrics")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1_score)
    print("Balanced Accuracy:", bal_acc)

    print("\nConfusion Matrix:")
    print("TP:", tp, "FP:", fp, "TN:", tn, "FN:", fn)


if __name__ == "__main__":
    regression()
    classification()

