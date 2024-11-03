import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import sys
import io
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

# Настройка кодировки для Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    import locale
    locale.setlocale(locale.LC_ALL, 'Russian_Russia.1251')

# Настройка стиля визуализации
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Получаем абсолютный путь к текущей директории
current_dir = os.path.dirname(os.path.abspath(__file__))

# Создаем директорию для сохранения графиков
pic_dir = os.path.join(current_dir, 'pic')
if not os.path.exists(pic_dir):
    os.makedirs(pic_dir)

# Загружаем данные с правильным путем
data_path = os.path.join(current_dir, 'data', 'user_behavior_dataset.csv')
df = pd.read_csv(data_path)

# Настройка отображения pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)

def analyze_target_distribution(data):
    """Анализ распределения целевой переменной"""
    plt.figure(figsize=(10, 6))
    target_dist = data['User Behavior Class'].value_counts().sort_index()
    sns.barplot(x=target_dist.index, y=target_dist.values)
    plt.title('Распределение классов поведения пользователей')
    plt.xlabel('Класс поведения')
    plt.ylabel('Количество пользователей')

    # Вывод процентного соотношения классов
    print("\nПроцентное соотношение классов:")
    print(data['User Behavior Class'].value_counts(normalize=True).sort_index().multiply(100).round(2))
    plt.savefig(os.path.join(pic_dir, 'target_distribution.png'))
    plt.close()

def analyze_numerical_features(data):
    """Анализ числовых признаков"""
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'User Behavior Class']

    for col in numerical_cols:
        plt.figure(figsize=(15, 5))

        # График распределения
        plt.subplot(1, 3, 1)
        sns.histplot(data=data, x=col, bins=30)
        plt.title(f'Распределение {col}')

        # Box plot по классам
        plt.subplot(1, 3, 2)
        sns.boxplot(data=data, x='User Behavior Class', y=col)
        plt.title(f'Box plot {col} по классам')

        # Violin plot
        plt.subplot(1, 3, 3)
        sns.violinplot(data=data, x='User Behavior Class', y=col)
        plt.title(f'Violin plot {col} по классам')

        plt.tight_layout()

        # Очищаем имя файла от специальных символов
        safe_filename = col.replace('/', '_').replace('\\', '_').replace(' ', '_')
        plt.savefig(os.path.join(pic_dir, f'{safe_filename}_analysis.png'))
        plt.close()

        # Статистические тесты
        print(f"\nСтатистика для {col}:")
        print(data[col].describe())

        # Тест на нормальность
        stat, p_value = stats.normaltest(data[col])
        print(f"Тест на нормальность (D'Agostino-Pearson): p-value = {p_value:.4f}")

def analyze_categorical_features(data):
    """Анализ категориальных признаков"""
    categorical_cols = data.select_dtypes(include=['object']).columns

    for col in categorical_cols:
        plt.figure(figsize=(12, 6))

        # Распределение категорий
        sns.countplot(data=data, x=col, hue='User Behavior Class')
        plt.xticks(rotation=45)
        plt.title(f'Распределение {col} по классам поведения')

        plt.tight_layout()
        plt.savefig(os.path.join(pic_dir, f'{col}_distribution.png'))
        plt.close()

        # Статистика
        print(f"\nСтатистика для {col}:")
        print(data[col].value_counts(normalize=True).multiply(100).round(2))

def correlation_analysis(data):
    """Анализ корреляй"""
    # Корреляционная матрица для числовых признаков
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = data[numerical_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Корреляционная матрица')
    plt.tight_layout()
    plt.savefig(os.path.join(pic_dir, 'correlation_matrix.png'))
    plt.close()

    # Топ коррелирующих признаков с целевой переменной
    target_corr = corr_matrix['User Behavior Class'].sort_values(ascending=False)
    print("\nКорреляция признаков с целевой переменной:")
    print(target_corr)

def initial_analysis(data):
    """Первичный анализ данных"""
    print("Общая информация о датасете:")
    print(data.info())
    print("\nПроверка на пропущенные значения:")
    print(data.isnull().sum())

    analyze_target_distribution(data)
    analyze_numerical_features(data)
    analyze_categorical_features(data)
    correlation_analysis(data)

def preprocess_data(data):
    """
    Предобработка данных:
    1. Удаление ненужных ризнаков
    2. Кодирование категориальных признаков
    3. Стандартизация числовых признаков
    """
    # Создаем копию данных
    df_processed = data.copy()

    # Удаляем User ID
    df_processed = df_processed.drop('User ID', axis=1)

    # Разделяем признаки на категориальные и числовые
    categorical_features = ['Device Model', 'Operating System', 'Gender']
    numerical_features = ['App Usage Time (min/day)', 'Screen On Time (hours/day)',
                         'Battery Drain (mAh/day)', 'Number of Apps Installed',
                         'Data Usage (MB/day)', 'Age']

    # Создаем преобразователи
    numeric_transformer = StandardScaler()

    # Для Operating System и Gender используем LabelEncoder
    le_os = LabelEncoder()
    le_gender = LabelEncoder()
    df_processed['Operating System'] = le_os.fit_transform(df_processed['Operating System'])
    df_processed['Gender'] = le_gender.fit_transform(df_processed['Gender'])

    # Для Device Model используем OneHotEncoder
    device_encoder = OneHotEncoder(sparse_output=False, drop='first')
    device_encoded = device_encoder.fit_transform(df_processed[['Device Model']])
    device_columns = [f"Device_{cat}" for cat in device_encoder.categories_[0][1:]]

    # Добавляем закодированные признаки устройств
    device_df = pd.DataFrame(device_encoded, columns=device_columns)
    df_processed = pd.concat([df_processed.drop('Device Model', axis=1), device_df], axis=1)

    # Стандартизируем числовые признаки
    scaler = StandardScaler()
    df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])

    # Сохраняем целвую переменную
    y = df_processed['User Behavior Class']
    X = df_processed.drop('User Behavior Class', axis=1)

    return X, y, scaler, le_os, le_gender, device_encoder

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Разделение данных на обучающую и тестовую выборки
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_baseline_model(X_train, X_test, y_train, y_test):
    """Обучение базовой модели (Логистическая регрессия)"""
    # Создаем и обучаем модель
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    # Предсказания
    y_pred = lr_model.predict(X_test)

    # Оценка модели
    print("\nLogistic Regression Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Построение confusion matrix
    plot_confusion_matrix(y_test, y_pred, "Логистическая регрессия")

    return lr_model

def train_random_forest(X_train, X_test, y_train, y_test):
    """Обучение Random Forest с GridSearch"""
    # Параметры для поиска
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Лучшая модель
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    print("\nRandom Forest Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Построение confusion matrix
    plot_confusion_matrix(y_test, y_pred, "Random Forest")

    return best_rf

def train_svm(X_train, X_test, y_train, y_test):
    """Обучение SVM с GridSearch"""
    # Параметры для поиска
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }

    svm_model = SVC(random_state=42)
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Лучшая модель
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test)

    print("\nSVM Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Построение confusion matrix
    plot_confusion_matrix(y_test, y_pred, "SVM")

    return best_svm

def plot_confusion_matrix(y_true, y_pred, title):
    """Построение confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(pic_dir, f'confusion_matrix_{title}.png'))
    plt.close()

def evaluate_models_cv(X, y, models_dict):
    """Оценка моделей с помощью cross-validation"""
    results = {}
    for name, model in models_dict.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        results[name] = {
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std()
        }
        print(f"\n{name} CV Results:")
        print(f"Mean Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    return results

def check_data_leakage(X, y):
    """Проверка на утечку данных"""
    correlations = X.corrwith(y)
    high_corr = correlations[correlations.abs() > 0.95]
    print("\nВысокая корреляция с целевой переменной:")
    print(high_corr)

def train_regularized_logistic_regression(X_train, X_test, y_train, y_test):
    """Обучение логистической регрессии с L2 регуляризацией"""
    lr_model = LogisticRegressionCV(cv=5, penalty='l2', max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    # Предсказания
    y_pred = lr_model.predict(X_test)

    # Оценка модели
    print("\nRegularized Logistic Regression Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Построение confusion matrix
    plot_confusion_matrix(y_test, y_pred, "Логистическая регрессия с регуляризацией")

    return lr_model

def plot_roc_curve_multiclass(y_true, y_score, title, n_classes):
    """Построение ROC-кривой для многоклассовой классификации"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-кривая - {title}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(pic_dir, f'roc_curve_{title}.png'))
    plt.close()

def evaluate_model_performance(model, X_test, y_test, title):
    """Оценка производительности модели"""
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test) if hasattr(model, "predict_proba") else model.decision_function(X_test)

    print(f"\n{title} Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Построение confusion matrix
    plot_confusion_matrix(y_test, y_pred, title)

    # Построение ROC-кривой для многоклассовой классификации
    plot_roc_curve_multiclass(y_test, y_score, title, n_classes=len(np.unique(y_test)))

if __name__ == "__main__":
    # Проводим первичный анализ данных
    initial_analysis(df)

    # Предобработка данных
    X, y, scaler, le_os, le_gender, device_encoder = preprocess_data(df)

    # Проверка на утечку данных
    check_data_leakage(X, y)

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Обучение регуляризованной логистической регрессии
    print("\nОбучение регуляризованной логистической регрессии...")
    reg_lr_model = train_regularized_logistic_regression(X_train, X_test, y_train, y_test)

    # Обучение Random Forest
    print("\nОбучение Random Forest...")
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)

    # Обучение SVM
    print("\nОбучение SVM...")
    svm_model = train_svm(X_train, X_test, y_train, y_test)

    # Оценка моделей
    evaluate_model_performance(reg_lr_model, X_test, y_test, "Regularized Logistic Regression")
    evaluate_model_performance(rf_model, X_test, y_test, "Random Forest")
    evaluate_model_performance(svm_model, X_test, y_test, "SVM")
