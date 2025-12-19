"""
Application Streamlit - Assistant Financier Intelligent
2 modes: Formulaire classique + Chatbot conversationnel
"""

import streamlit as st
import sys
import os
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from utils.prediction_engine import FinancialPredictor
    from utils.llm_mistral_api import FinancialChatbot
except ImportError:
    # Fallback pour Streamlit local
    st.error("Assurez-vous que les chemins vers 'prediction_engine.py' et 'llm_mistral_api.py' sont corrects.")
    sys.exit()

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================

st.set_page_config(
    page_title="Assistant Financier IA",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #1E88E5; 
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Les messages de chat seront gérés par st.chat_message (nouveau) */
</style>
""", unsafe_allow_html=True)


# ============================================================================
# INITIALISATION DES MODÈLES (Cache)
# ============================================================================

@st.cache_resource
def load_models():
    """Charge les modèles une seule fois au démarrage"""
    try:
        predictor = FinancialPredictor(
            model_path='model/best_financial_model_lightgbm_optimized.pkl',
            metadata_path='model/model_metadata_lightgbm_optimized.json'
        )
        chatbot = FinancialChatbot()
        return predictor, chatbot, None
    except Exception as e:
        return None, None, f"Erreur de chargement: {str(e)}"


# ============================================================================
# FONCTIONS D'AFFICHAGE
# ============================================================================

def display_prediction_card(prediction_result):
    """Affiche les résultats de prédiction dans une carte visuelle"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Prédiction</h3>
            <h1>{prediction_result['prediction']:.2f}€</h1>
            <p>Dépense prévue</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        ci_low, ci_high = prediction_result['confidence_interval']
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Intervalle</h3>
            <h2>{ci_low:.0f}€ - {ci_high:.0f}€</h2>
            <p>Fourchette de confiance</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>👤 Profil</h3>
            <h2>{prediction_result['profil_detecte']}</h2>
            <p>Niveau: {prediction_result['tier_conso']}</p>
        </div>
        """, unsafe_allow_html=True)


def display_gauge_chart(prediction, salaire, loyer):
    """Crée un graphique en jauge pour visualiser le budget"""
    
    budget_restant = salaire - loyer - prediction
    ratio = (prediction / salaire) * 100 if salaire > 0 else 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ratio,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "% du salaire dépensé"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1E88E5"}, 
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def display_budget_breakdown(salaire, loyer, prediction):
    """Graphique camembert du budget"""
    
    epargne = max(0, salaire - loyer - prediction)
    
    # Évite un camembert vide si le budget est nul ou négatif
    if salaire <= 0:
        st.warning("Salaire nul, impossible d'afficher la répartition.")
        return
        
    fig = go.Figure(data=[go.Pie(
        labels=['Loyer', 'Dépenses prévues', 'Épargne possible'],
        values=[loyer, prediction, epargne],
        hole=0.4,
        marker=dict(colors=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    )])
    
    fig.update_layout(
        title="Répartition du budget",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MODE 1 : FORMULAIRE CLASSIQUE
# ============================================================================

def formulaire_mode(predictor):
    """Interface formulaire pour saisie manuelle des données"""
    
    st.markdown("### Renseignez vos informations")
    
    # Utilisation de st.container pour la structure visuelle
    with st.container(border=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 💵 Revenus & Charges")
            salaire = st.number_input("Salaire mensuel net (€)", min_value=0, value=2500, step=100, key="f_salaire")
            loyer = st.number_input("Loyer mensuel (€)", min_value=0, value=800, step=50, key="f_loyer")
            nb_enfants = st.number_input("Nombre d'enfants", min_value=0, max_value=10, value=0, step=1, key="f_enfants")
        
        with col2:
            st.markdown("#### 💳 Dépenses récentes")
            depense_totale = st.number_input("Dépenses du mois en cours (€)", min_value=0, value=1200, step=50, key="f_dep_tot")
            depense_prev = st.number_input("Dépenses du mois précédent (€)", min_value=0, value=1150, step=50, key="f_dep_prev")
    
    st.markdown("---")
    
    # Bouton de prédiction
    if st.button("Prédire mes dépenses du mois prochain", type="primary", use_container_width=True):
        
        with st.spinner("⚙️ Analyse en cours..."):
            user_data = {
                'depense_totale': depense_totale,
                'depense_prev': depense_prev,
                'salaire': salaire,
                'loyer': loyer,
                'nb_enfants': nb_enfants
            }
            result = predictor.predict(user_data)
            st.session_state['last_prediction'] = result
            st.session_state['last_user_data'] = user_data
        
        st.success("Prédiction terminée !")
        st.balloons()
    
    # Affichage des résultats si disponibles
    if 'last_prediction' in st.session_state:
        st.markdown("---")
        st.markdown("## 📊 Résultats de l'analyse")
        
        result = st.session_state['last_prediction']
        user_data = st.session_state['last_user_data']
        
        display_prediction_card(result)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            display_gauge_chart(result['prediction'], user_data['salaire'], user_data['loyer'])
        with col2:
            display_budget_breakdown(user_data['salaire'], user_data['loyer'], result['prediction'])
        
        # Utilisation de st.expander pour les conseils
        st.markdown("---")
        st.markdown("## Conseils personnalisés")
        with st.expander("Cliquez ici pour lire vos conseils", expanded=True):
            for conseil in result['conseils']:
                st.info(f"✨ {conseil}")


# ============================================================================
# MODE 2 : CHATBOT CONVERSATIONNEL
# ============================================================================

def chatbot_mode(predictor, chatbot):
    """Interface chatbot avec historique de conversation"""
    
    st.markdown("### 💬 Posez votre question en langage naturel")
    
    # Initialisation de l'historique de chat (messages en format Streamlit/Dict)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [] # Liste de dicts {'role': 'user'/'assistant', 'content': str, 'timestamp': datetime}
    
    # Affichage de l'historique (Utilisation de st.chat_message)
    for message in st.session_state.chat_history:
        # st.chat_message gère le style user/assistant
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            
            # Affichage des détails si disponibles (prédiction du bot)
            if message['role'] == 'assistant' and 'prediction_data' in message:
                with st.expander("📊 Voir les détails de la prédiction"):
                    pred = message['prediction_data']
                    
                    st.markdown(f"**Profil détecté**: {pred['profil_detecte']} ({pred['tier_conso']})")
                    st.markdown(f"**Intervalle de confiance**: {pred['confidence_interval'][0]}€ - {pred['confidence_interval'][1]}€")
                    st.json(pred['features_utilisees'])
    
    # Zone de saisie et logique d'envoi
    if user_input := st.chat_input(
        placeholder="Ex: Je gagne 2500€, j'ai dépensé 1200€ ce mois. Combien vais-je dépenser le mois prochain ?",
    ):
        
        # Ajout du message utilisateur à l'historique
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now()
        })
        
        # Affichage immédiat du message utilisateur
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Traitement de la requête
        with st.chat_message("assistant"):
            with st.spinner("🤔 L'assistant réfléchit..."):
                try:
                    # Appel du pipeline LLM complet avec l'historique
                    # On envoie l'historique SANS le dernier message utilisateur qu'on vient d'ajouter
                    result = chatbot.process_user_query(
                        user_input, 
                        predictor, 
                        st.session_state.chat_history[:-1]
                    )
                    
                    # Affichage de la réponse
                    st.markdown(result['response'])
                    
                    # Affichage des détails de la prédiction
                    with st.expander("📊 Voir les détails de la prédiction"):
                        pred = result['prediction']
                        st.markdown(f"**Profil détecté**: {pred['profil_detecte']} ({pred['tier_conso']})")
                        st.markdown(f"**Intervalle de confiance**: {pred['confidence_interval'][0]}€ - {pred['confidence_interval'][1]}€")
                        st.json(pred['features_utilisees'])
                    
                    # Ajout de la réponse du bot à l'historique
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': result['response'],
                        'prediction_data': result['prediction'],
                        'timestamp': datetime.now()
                    })
                    
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
                    # Retirer le message utilisateur qui a causé l'erreur
                    st.session_state.chat_history.pop()
    
    # Bouton d'effacement
    st.markdown("---")
    if st.button("Effacer la conversation", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

# ============================================================================
# APPLICATION PRINCIPALE
# ============================================================================

def main():
    
    # Header
    st.markdown('<h1 class="main-header">💰 Assistant Financier Intelligent</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #666; font-size: 1.1rem;">
        Prédisez vos dépenses du mois prochain et obtenez des conseils personnalisés.
    </p>
    """, unsafe_allow_html=True)
    
    # Chargement des modèles
    with st.spinner("⚙️ Chargement des modèles..."):
        predictor, chatbot, error = load_models()
    
    if error:
        st.error(f"Erreur lors du chargement: {error}")
        st.info("Vérifiez que les chemins dans `load_models` (best_financial_model_...) et la `MISTRAL_API_KEY` sont corrects.")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2920/2920349.png", width=100)
        st.markdown("## ⚙️ Options")
        
        mode = st.radio("Choisissez un mode:", ["Formulaire", "Chatbot IA"], key="app_mode")
        
        st.markdown("---")
        st.markdown("### À propos du modèle")
        if predictor:
            st.info(f"""
            **Type**: {predictor.metadata['model_type'].upper()}
            **MAE (Test Set)**: {predictor.official_metrics['mae']:.2f}€
            **RMSE (Test Set)**: {predictor.official_metrics['rmse']:.2f}€
            """)
        
        st.markdown("---")
        st.caption("Développé pour un projet de Machine Learning & LLM.")
    
    # Affichage du mode sélectionné
    if st.session_state.app_mode == "Formulaire":
        formulaire_mode(predictor)
    else:
        chatbot_mode(predictor, chatbot)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; padding: 20px;">
        Propulsé par LightGBM et Mistral AI
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()