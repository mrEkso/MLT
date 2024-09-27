import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_curve, \
    precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Завантаження даних
data = pd.read_csv('./bioresponse.csv')

# Розподіл на змінні та цільове значення
X = data.drop(columns=['Activity'])
y = data['Activity']

# Розподіл на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. Дрібне дерево рішень
small_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
small_tree.fit(X_train, y_train)
y_pred_small_tree = small_tree.predict(X_test)

# 2. Глибоке дерево рішень
deep_tree = DecisionTreeClassifier(max_depth=None, random_state=42)
deep_tree.fit(X_train, y_train)
y_pred_deep_tree = deep_tree.predict(X_test)

# 3. Випадковий ліс на дрібних деревах
small_forest = RandomForestClassifier(max_depth=3, random_state=42)
small_forest.fit(X_train, y_train)
y_pred_small_forest = small_forest.predict(X_test)

# 4. Випадковий ліс на глибоких деревах
deep_forest = RandomForestClassifier(max_depth=None, random_state=42)
deep_forest.fit(X_train, y_train)
y_pred_deep_forest = deep_forest.predict(X_test)


# Функція для оцінки моделей
def evaluate_model(y_test, y_pred, model_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred)
    print(f"{model_name}:")
    print(
        f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Log-Loss: {loss:.4f}")
    print()


# Оцінка моделей
evaluate_model(y_test, y_pred_small_tree, "Дрібне дерево рішень")
evaluate_model(y_test, y_pred_deep_tree, "Глибоке дерево рішень")
evaluate_model(y_test, y_pred_small_forest, "Випадковий ліс на дрібних деревах")
evaluate_model(y_test, y_pred_deep_forest, "Випадковий ліс на глибоких деревах")


# Класифікатор для уникнення помилок II роду
def minimize_false_negatives(model, X_test, y_test, threshold=0.3):
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Зміна порогу для уникнення помилок II роду (False Negatives)
    y_pred_new = np.where(y_pred_prob > threshold, 1, 0)

    # Оцінка метрик
    evaluate_model(y_test, y_pred_new, f"Класифікатор з порогом {threshold}")


# Виклик функції для класифікатора, який уникає помилок II роду
minimize_false_negatives(deep_forest, X_test, y_test, threshold=0.3)


def plot_curves(models, X_test, y_test):
    # Побудова Precision-Recall кривих
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)  # Графік для Precision-Recall
    for model_name, model in models.items():
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{model_name} (PR-AUC = {pr_auc:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall криві')
    plt.legend(loc='best')

    # Побудова ROC-кривих
    plt.subplot(1, 2, 2)  # Графік для ROC
    for model_name, model in models.items():
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (ROC-AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')  # Діагональна лінія
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-криві')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()


# Список моделей для побудови
models = {
    "Дрібне дерево рішень": small_tree,
    "Глибоке дерево рішень": deep_tree,
    "Випадковий ліс на дрібних деревах": small_forest,
    "Випадковий ліс на глибоких деревах": deep_forest
}

# Виклик функції для побудови графіків
plot_curves(models, X_test, y_test)
