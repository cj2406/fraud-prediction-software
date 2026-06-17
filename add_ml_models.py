import json

# Read the existing notebook
with open('analysismodel.ipynb', 'r') as f:
    nb = json.load(f)

# New cells to add
new_cells = [
    {
        "cell_type": "markdown",
        "id": "ml-section",
        "metadata": {},
        "source": ["# Machine Learning Models\n", "Building predictive models for fraud detection"]
    },
    {
        "cell_type": "code",
        "id": "feature-prep",
        "metadata": {},
        "source": [
            "# Feature engineering and preparation\n",
            "from sklearn.preprocessing import LabelEncoder\n",
            "\n",
            "# Create a copy for modeling\n",
            "df_model = df.copy()\n",
            "\n",
            "# Encode categorical variables\n",
            "le_type = LabelEncoder()\n",
            "df_model['type_encoded'] = le_type.fit_transform(df_model['type'])\n",
            "\n",
            "# Select features for the model\n",
            "feature_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', \n",
            "                'newbalanceDest', 'type_encoded', 'balanceDiffOrig', 'balanceDiffDest']\n",
            "\n",
            "X = df_model[feature_cols]\n",
            "y = df_model['isFraud']\n",
            "\n",
            "print(f'Features shape: {X.shape}')\n",
            "print(f'Target distribution:\\n{y.value_counts()}')\n",
            "print(f'\\nFeature columns: {feature_cols}')"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "cell_type": "code",
        "id": "train-test-split",
        "metadata": {},
        "source": [
            "# Train-test split with stratification\n",
            "X_train, X_test, y_train, y_test = train_test_split(\n",
            "    X, y, test_size=0.3, random_state=42, stratify=y\n",
            ")\n",
            "\n",
            "print(f'Training set size: {X_train.shape[0]}')\n",
            "print(f'Test set size: {X_test.shape[0]}')\n",
            "print(f'\\nTraining fraud rate: {y_train.sum()/len(y_train)*100:.2f}%')\n",
            "print(f'Test fraud rate: {y_test.sum()/len(y_test)*100:.2f}%')"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "cell_type": "code",
        "id": "scale-features",
        "metadata": {},
        "source": [
            "# Scale features\n",
            "scaler = StandardScaler()\n",
            "X_train_scaled = scaler.fit_transform(X_train)\n",
            "X_test_scaled = scaler.transform(X_test)\n",
            "\n",
            "print('Features scaled successfully')\n",
            "print(f'Train mean: {X_train_scaled.mean(axis=0).round(3)}')\n",
            "print(f'Train std: {X_train_scaled.std(axis=0).round(3)}')"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "cell_type": "code",
        "id": "logistic-regression",
        "metadata": {},
        "source": [
            "from sklearn.metrics import roc_auc_score, roc_curve\n",
            "\n",
            "# Logistic Regression with class weights\n",
            "lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)\n",
            "lr.fit(X_train_scaled, y_train)\n",
            "\n",
            "# Predictions\n",
            "y_pred_lr = lr.predict(X_test_scaled)\n",
            "y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]\n",
            "\n",
            "# Evaluation\n",
            "auc_lr = roc_auc_score(y_test, y_pred_proba_lr)\n",
            "print('Logistic Regression Results:')\n",
            "print(f'AUC-ROC: {auc_lr:.4f}')\n",
            "print('\\nClassification Report:')\n",
            "print(classification_report(y_test, y_pred_lr))\n",
            "print('\\nConfusion Matrix:')\n",
            "print(confusion_matrix(y_test, y_pred_lr))"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "cell_type": "code",
        "id": "random-forest",
        "metadata": {},
        "source": [
            "from sklearn.ensemble import RandomForestClassifier\n",
            "\n",
            "# Random Forest\n",
            "rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42,\n",
            "                            class_weight='balanced', n_jobs=-1)\n",
            "rf.fit(X_train, y_train)\n",
            "\n",
            "# Predictions\n",
            "y_pred_rf = rf.predict(X_test)\n",
            "y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]\n",
            "\n",
            "# Evaluation\n",
            "auc_rf = roc_auc_score(y_test, y_pred_proba_rf)\n",
            "print('Random Forest Results:')\n",
            "print(f'AUC-ROC: {auc_rf:.4f}')\n",
            "print('\\nClassification Report:')\n",
            "print(classification_report(y_test, y_pred_rf))\n",
            "\n",
            "# Feature importance\n",
            "importance_df = pd.DataFrame({\n",
            "    'feature': feature_cols,\n",
            "    'importance': rf.feature_importances_\n",
            "}).sort_values('importance', ascending=False)\n",
            "\n",
            "print('\\nFeature Importance:')\n",
            "print(importance_df)"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "cell_type": "code",
        "id": "feature-importance-plot",
        "metadata": {},
        "source": [
            "# Feature importance visualization\n",
            "plt.figure(figsize=(10, 6))\n",
            "plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')\n",
            "plt.xlabel('Importance')\n",
            "plt.title('Random Forest Feature Importance')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "cell_type": "code",
        "id": "roc-curves",
        "metadata": {},
        "source": [
            "# ROC curves comparison\n",
            "fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)\n",
            "fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)\n",
            "\n",
            "plt.figure(figsize=(10, 6))\n",
            "plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={auc_lr:.4f})', linewidth=2)\n",
            "plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.4f})', linewidth=2)\n",
            "plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')\n",
            "plt.xlabel('False Positive Rate')\n",
            "plt.ylabel('True Positive Rate')\n",
            "plt.title('ROC Curves - Model Comparison')\n",
            "plt.legend()\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.show()"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "cell_type": "code",
        "id": "confusion-matrices",
        "metadata": {},
        "source": [
            "from sklearn.metrics import ConfusionMatrixDisplay\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
            "\n",
            "# Logistic Regression\n",
            "ConfusionMatrixDisplay(\n",
            "    confusion_matrix(y_test, y_pred_lr),\n",
            "    display_labels=['Legit', 'Fraud']\n",
            ").plot(ax=axes[0], cmap='Blues')\n",
            "axes[0].set_title('Logistic Regression')\n",
            "\n",
            "# Random Forest\n",
            "ConfusionMatrixDisplay(\n",
            "    confusion_matrix(y_test, y_pred_rf),\n",
            "    display_labels=['Legit', 'Fraud']\n",
            ").plot(ax=axes[1], cmap='Greens')\n",
            "axes[1].set_title('Random Forest')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "cell_type": "code",
        "id": "model-comparison",
        "metadata": {},
        "source": [
            "from sklearn.metrics import precision_recall_fscore_support\n",
            "\n",
            "# Model comparison\n",
            "metrics_lr = precision_recall_fscore_support(y_test, y_pred_lr, average='binary')\n",
            "metrics_rf = precision_recall_fscore_support(y_test, y_pred_rf, average='binary')\n",
            "\n",
            "comparison = pd.DataFrame({\n",
            "    'Logistic Regression': {\n",
            "        'Precision': metrics_lr[0],\n",
            "        'Recall': metrics_lr[1],\n",
            "        'F1-Score': metrics_lr[2],\n",
            "        'AUC-ROC': auc_lr\n",
            "    },\n",
            "    'Random Forest': {\n",
            "        'Precision': metrics_rf[0],\n",
            "        'Recall': metrics_rf[1],\n",
            "        'F1-Score': metrics_rf[2],\n",
            "        'AUC-ROC': auc_rf\n",
            "    }\n",
            "})\n",
            "\n",
            "print('Model Comparison:')\n",
            "print(comparison.round(4))\n",
            "print(f'\\nBest Model: Random Forest (AUC: {auc_rf:.4f})')"
        ],
        "outputs": [],
        "execution_count": None
    }
]

# Append new cells to notebook
nb['cells'].extend(new_cells)

# Write updated notebook
with open('analysismodel.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Added ML model cells to notebook")
