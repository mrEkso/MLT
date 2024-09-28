import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo  # Імпорт бібліотеки для завантаження даних

# Завантаження набору даних з UCI
banknote_authentication = fetch_ucirepo(id=267)

# Отримуємо ознаки (X) та мітки класів (y)
X = banknote_authentication.data.features
y = banknote_authentication.data.targets

# Масштабуємо дані
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Розділяємо дані на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 1. Логістична регресія з Logistic Loss
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# 2. AdaBoost з Adaboost loss
adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost.fit(X_train, y_train)

# 3. Логістична регресія з Binary Crossentropy
# Примітка: у sklearn логістична регресія вже використовує Binary Crossentropy
log_reg_bce = LogisticRegression()
log_reg_bce.fit(X_train, y_train)

# Обчислимо функції втрат для кожної моделі
log_loss_train = log_loss(y_train, log_reg.predict_proba(X_train))
log_loss_test = log_loss(y_test, log_reg.predict_proba(X_test))

adaboost_loss_train = log_loss(y_train, adaboost.predict_proba(X_train))
adaboost_loss_test = log_loss(y_test, adaboost.predict_proba(X_test))

bce_loss_train = log_loss(y_train, log_reg_bce.predict_proba(X_train))
bce_loss_test = log_loss(y_test, log_reg_bce.predict_proba(X_test))

# Виведемо точність (accuracy) для кожної моделі
accuracy_log_reg = accuracy_score(y_test, log_reg.predict(X_test))
accuracy_adaboost = accuracy_score(y_test, adaboost.predict(X_test))
accuracy_bce = accuracy_score(y_test, log_reg_bce.predict(X_test))

print(f"Accuracy (Logistic Loss): {accuracy_log_reg}")
print(f"Accuracy (Adaboost Loss): {accuracy_adaboost}")
print(f"Accuracy (Binary Crossentropy): {accuracy_bce}")

# Візуалізація кривих навчання для кожної функції втрат
epochs = ['Train', 'Test']
loss_values = {
    'Logistic Loss': [log_loss_train, log_loss_test],
    'Adaboost Loss': [adaboost_loss_train, adaboost_loss_test],
    'Binary Crossentropy': [bce_loss_train, bce_loss_test],
}

plt.figure(figsize=(10, 6))
for key, values in loss_values.items():
    plt.plot(epochs, values, marker='o', label=key)

plt.title('Криві навчання для різних функцій втрат')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

