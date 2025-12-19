# ==============================================================================
# 1. CONFIGURATION AVANCÉE & SEGMENTATION
# ==============================================================================
from faker import Faker
import random
import numpy as np
import datetime
from datetime import datetime, timedelta
import pandas as pd
fake = Faker('fr_FR')
random.seed(42)
np.random.seed(42)

nb_users = 1500
start_date = datetime(2020, 1, 1)
end_date = datetime(2025, 1, 31)

# Segmentation des marchands par coût (Eco / Standard / Premium)
MARCHANDS_TIERS = {
    'Alimentation': {
        'Eco': ['Lidl', 'Aldi', 'Netto', 'Leclerc'],
        'Standard': ['Carrefour', 'Auchan', 'Intermarché', 'Super U'],
        'Premium': ['Monoprix', 'Biocoop', 'Grand Frais', 'La Grande Épicerie']
    },
    'Shopping': {
        'Eco': ['Vinted', 'Action', 'Primark', 'Kiabi'],
        'Standard': ['H&M', 'Zara', 'Decathlon', 'Amazon'],
        'Premium': ['Galeries Lafayette', 'Printemps', 'Boutique Indépendante', 'Apple Store']
    }
}

# Les autres restent génériques
MARCHANDS_GENERIC = {
    'Restaurant': ['McDonald\'s', 'Bistrot du Coin', 'UberEats', 'Sushi Shop', 'Brasserie'],
    'Transport': ['SNCF', 'RATP', 'Total Access', 'Uber', 'Shell'],
    'Loisirs': ['Netflix', 'UGC Cinéma', 'Basic Fit', 'Spotify', 'Fnac', 'Bar PMU'],
    'Santé': ['Pharmacie', 'Doctolib Consultation', 'Laboratoire'],
    'Enfance': ['Cantine Scolaire', 'Crèche Municipale', 'Orchestra'],
    'Logement': ['Loyer', 'Foncia', 'Nexity'],
    'Services Publics': ['EDF', 'Engie', 'Free Mobile', 'Orange'],
    'Aides': ['CAF Virement', 'CPAM Remboursement'],
    'Imprévus': ['Garage Auto', 'Plombier', 'Dentiste (Non remboursé)', 'Amende Stationnement']
}

# ==============================================================================
# 2. LOGIQUE COMPORTEMENTALE
# ==============================================================================

def get_merchant_tier(salaire):
    """Définit où l'utilisateur fait ses courses selon ses revenus."""
    if salaire < 1500: return 'Eco'
    elif salaire > 3500: return 'Premium'
    else: return 'Standard'

def pick_merchant(categorie, tier):
    """Choisit un marchand cohérent avec le niveau de vie."""
    if categorie in MARCHANDS_TIERS:
        # 80% de chance d'aller dans sa gamme, 20% de mixer
        if random.random() < 0.8:
            return random.choice(MARCHANDS_TIERS[categorie][tier])
        else:
            # Parfois un riche va chez Lidl, ou un étudiant se fait plaisir chez Monoprix
            all_merchants = [m for sublist in MARCHANDS_TIERS[categorie].values() for m in sublist]
            return random.choice(all_merchants)
    else:
        return random.choice(MARCHANDS_GENERIC.get(categorie, ['Marchand Inconnu']))

def create_smart_profile():
    """Crée un profil avec un 'point de rupture' financier."""
    # ... (Logique de profil précédente simplifiée pour l'exemple) ...
    profil_type = random.choices(
        ['Étudiant', 'Jeune actif', 'Cadre', 'Famille Modeste', 'Famille Aisée', 'Retraité'],
        weights=[0.15, 0.25, 0.20, 0.20, 0.10, 0.10]
    )[0]
    
    base_salaires = {
        'Étudiant': (800, 1200), 'Jeune actif': (1600, 2400), 'Cadre': (3000, 6000),
        'Famille Modeste': (2000, 3000), 'Famille Aisée': (4500, 8000), 'Retraité': (1400, 2500)
    }
    
    salaire = random.randint(*base_salaires[profil_type])
    tier_conso = get_merchant_tier(salaire)
    
    # Nombre d'enfants
    nb_enfants = 0
    if 'Famille' in profil_type: nb_enfants = random.randint(1, 4)
    elif profil_type == 'Cadre' and random.random() < 0.5: nb_enfants = random.randint(1, 3)
    
    # Loyer
    loyer = int(salaire * random.uniform(0.25, 0.45))
    
    return {
        'profil': profil_type, 'salaire': salaire, 'loyer': loyer,
        'nb_enfants': nb_enfants, 'tier_conso': tier_conso,
        # Facteur de consommation (Taille du foyer)
        'conso_factor': 1 + (0.3 * nb_enfants)
    }

# ==============================================================================
# 3. CŒUR DU SYSTÈME : BOUCLE AVEC RÉTROACTION (FEEDBACK LOOP)
# ==============================================================================

def generate_realistic_history(user_id, user_profile):
    transactions = []
    current_date = start_date
    
    # Solde initial (Un peu d'épargne ou 0)
    solde_actuel = random.randint(0, user_profile['salaire'])
    
    # Seuil de panique : Quand le solde passe sous ce montant, on arrête les frais
    seuil_alerte = 100 if user_profile['profil'] == 'Étudiant' else 300
    
    # Aides sociales (CAF/APL)
    montant_caf = 0
    if user_profile['nb_enfants'] >= 2: montant_caf += 142 + (user_profile['nb_enfants']-2)*180
    if user_profile['salaire'] < 1500: montant_caf += 150 # APL estimée
    
    while current_date <= end_date:
        is_weekend = current_date.weekday() >= 5
        jour = current_date.day
        mois = current_date.month
        daily_transactions = []

        # --- A. ENTRÉES D'ARGENT (Revenus) ---
        if jour == 1: # Salaire
            montant = user_profile['salaire']
            daily_transactions.append({'type': 'credit', 'cat': 'Salaire', 'montant': montant, 'marchand': 'Employeur'})
            solde_actuel += montant
            
        if jour == 5 and montant_caf > 0: # CAF
            daily_transactions.append({'type': 'credit', 'cat': 'Aides Sociales', 'montant': montant_caf, 'marchand': 'CAF'})
            solde_actuel += montant_caf

        # --- B. SORTIES FIXES (Obligatoires) ---
        # Ces dépenses passent MÊME si on est à découvert
        if jour == 4: # Loyer
            daily_transactions.append({'type': 'debit', 'cat': 'Logement', 'montant': user_profile['loyer'], 'marchand': 'Loyer'})
            solde_actuel -= user_profile['loyer']
            
        if jour == 10: # Services (Energie/Tel)
            montant = 60 * user_profile['conso_factor']
            daily_transactions.append({'type': 'debit', 'cat': 'Services Publics', 'montant': montant, 'marchand': 'EDF/Tel'})
            solde_actuel -= montant

        # --- C. DÉPENSES VARIABLES (Régulées par le solde) ---
        
        # 1. Calcul du "Stress Financier"
        # Si solde bas, probabilité de dépense divisée par 10
        propension_depense = 1.0
        if solde_actuel < seuil_alerte:
            propension_depense = 0.1 # Mode survie activé
        elif solde_actuel < 0:
            propension_depense = 0.01 # Bloqué (sauf alimentaire vital)

        # 2. Le "Gros Plein" Hebdomadaire (Samedi)
        # Pour les familles, c'est rituel, pour les étudiants, c'est plus fragmenté
        fait_courses = False
        if user_profile['nb_enfants'] > 0 and current_date.weekday() == 5: # Samedi
            if random.random() < 0.9 * propension_depense: # Quasi sûr d'y aller sauf si banqueroute totale
                montant = np.random.normal(150, 30) * user_profile['conso_factor']
                marchand = pick_merchant('Alimentation', user_profile['tier_conso'])
                daily_transactions.append({'type': 'debit', 'cat': 'Alimentation', 'montant': montant, 'marchand': marchand})
                solde_actuel -= montant
                fait_courses = True # On a fait les grosses courses, on n'achète pas de bricoles aujourd'hui

        # 3. Autres dépenses quotidiennes (si pas fait de grosses courses)
        if not fait_courses:
            # Nombre de transactions : Poisson loi, ajustée par le stress financier
            nb_achats = np.random.poisson(1.2 * propension_depense)
            
            for _ in range(nb_achats):
                # Choix catégorie (Loisirs/Resto sautent en premier si stress)
                if solde_actuel < seuil_alerte:
                    # Si pauvre: on n'achète que de la bouffe ou transport
                    cats = ['Alimentation', 'Transport']
                    weights = [0.8, 0.2]
                else:
                    # Si riche: on se fait plaisir
                    cats = ['Alimentation', 'Restaurant', 'Loisirs', 'Shopping', 'Transport']
                    weights = [0.3, 0.15, 0.15, 0.25, 0.15]

                cat = random.choices(cats, weights=weights)[0]
                
                # Montant
                base_price = 20
                if cat == 'Shopping': base_price = 60
                montant = abs(np.random.normal(base_price, base_price*0.5)) * user_profile['conso_factor']
                
                # Marchand
                marchand = pick_merchant(cat, user_profile['tier_conso'])
                
                daily_transactions.append({'type': 'debit', 'cat': cat, 'montant': round(montant, 2), 'marchand': marchand})
                solde_actuel -= montant

        # --- D. IMPRÉVUS (Accidents de la vie) ---
        # 1% de chance par jour d'un pépin, indépendant du solde (ça tombe même si on est pauvre)
        if random.random() < 0.005: 
            montant_pepin = random.choice([150, 300, 600]) # Amende, Garage, Plombier
            daily_transactions.append({'type': 'debit', 'cat': 'Imprévus', 'montant': montant_pepin, 'marchand': random.choice(MARCHANDS_GENERIC['Imprévus'])})
            solde_actuel -= montant_pepin

        # Ajout des transactions du jour avec la date
        for t in daily_transactions:
            t.update({'date': current_date, 'user_id': user_id})
            transactions.append(t)

        current_date += timedelta(days=1)

    return transactions

# ==============================================================================
# 4. GÉNÉRATION & EXPORT
# ==============================================================================
print("⚡ Génération avec simulation de solde temps réel...")
all_data = []
user_registry = []

for uid in range(1, nb_users + 1):
    profil = create_smart_profile()
    user_registry.append({**profil, 'user_id': uid}) # Save profil info
    
    trans = generate_realistic_history(uid, profil)
    all_data.extend(trans)

df = pd.DataFrame(all_data)
df_users = pd.DataFrame(user_registry)

# --- PRÉPARATION PREDICTION (Code identique à avant, résumé) ---
# On agrège pour le ML
df['mois'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()
df_ml = df[df['type'] == 'debit'].groupby(['user_id', 'mois'])['montant'].sum().reset_index()
df_ml.rename(columns={'montant': 'depense_totale'}, inplace=True)

# Feature Engineering (Lag)
df_ml = df_ml.sort_values(['user_id', 'mois'])
df_ml['depense_prev'] = df_ml.groupby('user_id')['depense_totale'].shift(1)
df_ml['target'] = df_ml.groupby('user_id')['depense_totale'].shift(-1)

# Merge avec profils riches
df_final = pd.merge(df_ml, df_users, on='user_id').dropna()

print(f"Terminé ! {len(df)} transactions générées.")
print(df_final[['user_id', 'mois', 'profil', 'tier_conso', 'depense_totale']].head(10))

# Sauvegarde
df.to_csv('transactions.csv', index=False)
df_final.to_csv('dataset_ml.csv', index=False)