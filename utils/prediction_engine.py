"""
Moteur de prédiction intelligent pour l'assistant financier
Gère la reconstruction des features et l'inférence du modèle
"""

import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

class FinancialPredictor:
    """Classe qui gère toute la logique de prédiction"""
    
    def __init__(self, model_path: str = '../model/best_financial_model_lightgbm_optimized.pkl',
                 metadata_path: str = '../model/model_metadata_lightgbm_optimized.json'):
        """Charge le modèle et ses métadonnées"""
        
        # Nous ciblons explicitement les fichiers optimisés de LightGBM
        self.model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Les features sont celles qui ont servi à entraîner le modèle 
        self.features = self.metadata['features']
        
        # Utilisation des métriques du Test Set pour la performance officielle
        self.official_metrics = self.metadata['metrics_test']
        
        print(f"Modèle {self.metadata['model_type']} chargé avec succès!")
        print(f"   MAE (Test Set): {self.official_metrics['mae']:.2f}€")
        
        
    def infer_tier_from_salary(self, salaire: float) -> str:
        """Déduit le tier de consommation depuis le salaire"""
        if salaire < 1500:
            return 'Eco'
        elif salaire > 3500:
            return 'Premium'
        else:
            return 'Standard'
    
    def infer_profile(self, salaire: float, nb_enfants: int = 0) -> str:
        """Infère le profil socio-économique"""
        if salaire < 1300:
            return 'Étudiant'
        elif nb_enfants >= 2 and salaire > 4500:
            return 'Famille Aisée'
        elif nb_enfants >= 1 and salaire < 3500:
            return 'Famille Modeste'
        elif salaire > 3000:
            return 'Cadre'
        elif salaire > 1600 and salaire <= 3000:
            return 'Jeune actif'
        else:
            return 'Retraité'
    
    def encode_categorical(self, profil: str, tier: str) -> tuple:
        """
        Encode les variables catégorielles. 
        L'encodage correspond à l'ordre des cat.codes du training set.
        """
        
        # ORDRE DES PROFILS
        profil_map = {
            'Cadre': 0, 
            'Famille Aisée': 1,
            'Famille Modeste': 2,
            'Jeune actif': 3,
            'Retraité': 4,
            'Étudiant': 5
        }
        
        # ORDRE DES TIERS
        tier_map = {
            'Eco': 0, 
            'Premium': 1, 
            'Standard': 2
        }
        
        # Fallback si le profil n'est pas reconnu: Jeune actif (3) et Standard (2)
        profil_code = profil_map.get(profil, 3) 
        tier_code = tier_map.get(tier, 2)
        
        return profil_code, tier_code
    
    def create_temporal_features(self, mois: datetime = None) -> Dict[str, int]:
        """Génère les features temporelles pour le mois de prédiction (le mois prochain)"""
        if mois is None:
            # On prédit le mois prochain
            mois = datetime.now().replace(day=1) + pd.DateOffset(months=1)
        
        return {
            'month': mois.month,
            'quarter': (mois.month - 1) // 3 + 1,
            'periode_fetes': 1 if mois.month in [11, 12] else 0,
            'time_idx': 0 
        }
    
    def reconstruct_features(self, user_input: Dict[str, Any]) -> pd.DataFrame:
        """
        Reconstruit toutes les features attendues par le modèle (dans l'ordre et le format exact)
        """
    
        # Récupération des données utilisateur
        depense_totale = user_input.get('depense_totale', user_input.get('depense_prev', 1000) * 1.05)
        depense_prev = user_input.get('depense_prev', depense_totale * 0.95)
        salaire = user_input['salaire']
        loyer = user_input['loyer']
        nb_enfants = user_input.get('nb_enfants', 0)
    
        # Calcul des features Lag/Rolling
        # Si lag_1 est fourni, l'utiliser, sinon estimer
        lag_1_val = user_input.get('lag_1')
        if lag_1_val is None:
            lag_1_val = depense_prev 
    
        lag_2_val = user_input.get('lag_2')
        if lag_2_val is None:
            lag_2_val = depense_prev * 0.95 
    
        lag_3_val = user_input.get('lag_3')
        if lag_3_val is None:
            lag_3_val = depense_prev * 0.90  
    
        # Reconstitution de la Rolling Mean 3 (t-1, t-2, t-3)
        rolling_mean_3 = np.mean([depense_prev, lag_1_val, lag_2_val])
    
        # Inférence du profil
        tier = self.infer_tier_from_salary(salaire)
        profil = self.infer_profile(salaire, nb_enfants)
        profil_code, tier_code = self.encode_categorical(profil, tier)

        # Features temporelles
        temporal = self.create_temporal_features()
        
        # Reconstruction du dictionnaire complet
        features_dict = {
            'depense_totale': depense_totale,     # Dépense du mois t
            'depense_prev': depense_prev,         # Dépense du mois t-1
            'lag_1': lag_1_val,                   # Dépense du mois t-2
            'lag_2': lag_2_val,                   # Dépense du mois t-3
            'lag_3': lag_3_val,                   # Dépense du mois t-4
            'rolling_mean_3': rolling_mean_3,     # Moyenne (t-1, t-2, t-3)
            'salaire': salaire,
            'loyer': loyer,
            'nb_enfants': nb_enfants,
            'profil_code': profil_code,
            'tier_code': tier_code,
            'month': temporal['month'],
            'quarter': temporal['quarter'],
            'periode_fetes': temporal['periode_fetes'],
            'time_idx': 0
        }
    
        # Création de la DataFrame avec l'ordre exact attendu par self.features
        df = pd.DataFrame([features_dict])[self.features]
    
        # Validation de l'ordre et des types des features
        if len(df.columns) != len(self.features) or list(df.columns) != self.features:
            raise ValueError("L'ordre ou le nombre des features ne correspond pas aux métadonnées du modèle.")

        return df, profil, tier
    
    def predict(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prédit la dépense du mois prochain
        """
        
        # Reconstruction des features
        X, profil, tier = self.reconstruct_features(user_input)
        
        # Prédiction
        y_pred = self.model.predict(X)[0]
        
        # Intervalle de confiance (±1 RMSE du Test Set)
        rmse = self.official_metrics['rmse']
        lower = max(0, y_pred - rmse)
        upper = y_pred + rmse
        
        # Génération de conseils personnalisés
        conseils = self._generate_advice(user_input, y_pred, profil, tier)
        
        return {
            'prediction': round(y_pred, 2),
            'profil_detecte': profil,
            'tier_conso': tier,
            'confidence_interval': (round(lower, 2), round(upper, 2)),
            'conseils': conseils,
            'features_utilisees': X.to_dict('records')[0]
        }
    
    # Génération de conseils personnalisés
    def _generate_advice(self, user_input: Dict, prediction: float, 
                        profil: str, tier: str) -> List[str]:
        """Génère des conseils personnalisés"""
        conseils = []
        
        # Conseil 1: Tendance des dépenses
        depense_actuelle = user_input.get('depense_totale', user_input.get('depense_prev', 0))
        
        # Éviter la division par zéro si les dépenses sont nulles
        if depense_actuelle > 0:
            variation = ((prediction - depense_actuelle) / depense_actuelle) * 100
        else:
            variation = 0

        if variation > 10:
            conseils.append(f"Attention : Hausse prévue de {variation:.1f}% le mois prochain")
        elif variation < -10:
            conseils.append(f"Bonne nouvelle : Baisse prévue de {abs(variation):.1f}%")
        else:
            conseils.append(f"Stabilité attendue (variation de {variation:.1f}%)")
        
        # Conseil 2: Ratio loyer/salaire
        ratio_loyer = (user_input['loyer'] / user_input['salaire']) * 100 if user_input['salaire'] > 0 else 0
        if ratio_loyer > 35:
            conseils.append(f"Votre loyer représente {ratio_loyer:.0f}% de vos revenus (recommandé: <33%)")
        
        # Conseil 3: Épargne potentielle
        epargne_possible = user_input['salaire'] - prediction - user_input['loyer']
        if epargne_possible > 0:
            conseils.append(f"Épargne potentielle estimée : {epargne_possible:.0f}€")
        else:
            conseils.append(f"Risque de découvert : {abs(epargne_possible):.0f}€ (Revoyez vos dépenses)")
        
        # Conseil 4: Adaptation au profil
        if profil == 'Étudiant':
            conseils.append("Astuce étudiant : Privilégiez les enseignes discount et cuisinez maison")
        elif 'Famille' in profil:
            conseils.append("Conseil famille : Planifiez vos achats en gros et profitez des aides CAF")
        
        return conseils

