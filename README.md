# 💰 Assistant Financier IA : Prédiction de Budget & Chatbot Intelligent

Ce projet présente un écosystème complet d'intelligence artificielle conçu pour aider les utilisateurs à anticiper leurs dépenses mensuelles. Il combine un moteur de prédiction basé sur le Machine Learning (LightGBM) et un chatbot conversationnel alimenté par Mistral AI via LangChain.

## 🌟 Fonctionnalités Clés

* **Prédiction Précise** : Estimation des dépenses du mois prochain avec un intervalle de confiance basé sur le RMSE du modèle.
* **Chatbot Conversationnel** : Interface naturelle capable d'extraire des données financières complexes d'une conversation et de maintenir un historique de dialogue.
* **Analyse de Profil** : Détection automatique du profil socio-économique (Étudiant, Cadre, Famille, etc.) pour adapter les conseils.
* **Visualisations Interactives** : Tableaux de bord Plotly affichant la répartition du budget et des jauges de santé financière.

## Architecture du Projet

Le projet est structuré en trois phases majeures :

### 1. Analyse et Préparation des Données (`.ipynb`)

* **EDA** : Exploration d'un dataset de plus de 3 millions de transactions pour comprendre les corrélations entre revenus, loyers et profils.
* **Feature Engineering** : Création de variables de décalage (*Lags*), de moyennes mobiles (*Rolling Mean*) et de variables temporelles (trimestres, périodes de fêtes).
* **Encodage** : Transformation des variables catégorielles en codes numériques exploitables par les modèles.

### 2. Modélisation et Optimisation (`modeling.ipynb`)

* **Algorithme** : Utilisation de **LightGBM** pour sa rapidité et sa précision sur les données tabulaires.
* **Performance** :
* **MAE** : 198.60 € (Erreur moyenne d'environ 8 %).
* **R²** : 0.9452 (Le modèle explique 94,5 % de la variance des dépenses).
* **Régularisation** : Optimisation des hyperparamètres pour réduire l'overfitting à seulement 6 € d'écart entre l'entraînement et la validation.



### 3. Déploiement et Interface (`.py`)

* **`prediction_engine.py`** : Moteur d'inférence gérant la reconstruction des données en temps réel.
* **`llm_mistral_api.py`** : Pipeline LangChain pour l'extraction d'entités financières et la génération de réponses empathiques.
* **`financial_streamlit_app.py`** : Application web interactive permettant la saisie par formulaire ou par chat.

## Installation et Lancement

### Prérequis

* Python 3.9+
* Clé API Mistral AI

### Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/votre-repo/financial-assistant-ia.git
cd financial-assistant-ia

```


2. Installez les dépendances :
```bash
pip install -r requirements.txt

```


3. Configurez vos variables d'environnement dans un fichier `.env` :
```env
MISTRAL_API_KEY=votre_cle_api_ici

```



### Lancement

Exécutez l'application Streamlit :

```bash
streamlit run financial_streamlit_app.py

```

## Métriques de Performance du Modèle Final

| Métrique | Résultat sur Test Set |
| --- | --- |
| **MAE (Erreur Absolue Moyenne)** | 198.60 € |
| **RMSE (Erreur Quadratique)** | 283.98 € |
| **R² (Coefficient de Corrélation)** | 0.9452 |
| **MAPE (Erreur en %)** | 8.13 % |

## Technologies Utilisées

* **Langages** : Python (Pandas, Numpy, Scikit-Learn).
* **ML** : LightGBM, XGBoost, Joblib.
* **IA Générative** : Mistral AI, LangChain (Pydantic Output Parser).
* **Interface** : Streamlit, Plotly.

---

*Ce projet a été développé dans le cadre d'un module de Machine Learning & IA pour démontrer l'intégration de modèles prédictifs classiques avec les capacités modernes des LLM.*

---