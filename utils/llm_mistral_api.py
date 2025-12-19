"""
Intégration Mistral AI avec LangChain pour extraction d'informations financières
"""

import os
import json
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # Import AIMessage pour l'historique

# Charger les variables d'environnement
load_dotenv()

class FinancialInfo(BaseModel):
    """Modèle Pydantic pour les informations financières extraites"""
    
    depense_totale: float = Field(description="Dépense totale du mois en cours (en euros)", ge=0)
    depense_prev: float = Field(description="Dépense totale du mois précédent (en euros)", ge=0)
    salaire: float = Field(description="Salaire mensuel net de l'utilisateur (en euros)", ge=0)
    loyer: float = Field(description="Montant du loyer mensuel (en euros). Si non mentionné, estimer à 30% du salaire.", ge=0)
    nb_enfants: int = Field(default=0, description="Nombre d'enfants à charge. Par défaut 0.", ge=0)
    lag_1: Optional[float] = Field(default=None, description="Dépense il y a 1 mois (optionnel)")
    lag_2: Optional[float] = Field(default=None, description="Dépense il y a 2 mois (optionnel)")
    lag_3: Optional[float] = Field(default=None, description="Dépense il y a 3 mois (optionnel)")
    
    # Estimation du loyer si non fourni
    @validator('loyer', pre=True, always=True)
    def estimate_loyer(cls, v, values):
        if v is None or v == 0:
            salaire = values.get('salaire', 0)
            return salaire * 0.30
        return v

# ============================================================================
# ASSISTANT FINANCIER LLM
# ============================================================================

class FinancialChatbot:
    """Chatbot financier utilisant Mistral AI pour comprendre les demandes utilisateurs et maintenir le contexte"""
    
    def __init__(self, model_name: str = "mistral-large-latest"):
        
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY non trouvée dans le .env")
        
        self.llm = ChatMistralAI(
            model=model_name,
            temperature=0,
            mistral_api_key=api_key
        )
        self.parser = PydanticOutputParser(pydantic_object=FinancialInfo)
        
        # PROMPT D'EXTRACTION
        self.extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_message}")
        ])
        
        # PROMPT DE RÉPONSE CONVERSATIONNELLE
        self.response_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Tu es un assistant financier bienveillant et expert.
            Ton rôle: Expliquer la prédiction, donner des conseils personnalisés, être encourageant et constructif.
            Ne mentionne JAMAIS les aspects techniques (features, modèle ML, etc.)."""),
            
            # EMPLACEMENT POUR L'HISTORIQUE DU CHAT
            MessagesPlaceholder(variable_name="chat_history"),
            
            ("human", """L'utilisateur a dit: "{user_message}"
            
            Informations extraites: {extracted_info}
            
            Résultats de la prédiction:
            - Dépense prévue: {prediction}€
            - Intervalle de confiance: {confidence}
            - Profil détecté: {profil}
            - Conseils: {conseils}
            
            Réponds de manière naturelle et personnalisée en tenant compte de la conversation précédente.""")
        ])
        
        print(f"Chatbot initialisé avec {model_name}")

    def _get_system_prompt(self) -> str:
        """Génère le prompt système pour l'extraction d'informations financières"""
        format_instructions = self.parser.get_format_instructions()
        
        format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
    
        return f"""Tu es un assistant spécialisé dans l'extraction d'informations financières.

MISSION: Extraire les informations suivantes depuis le message utilisateur et les retourner au format JSON.

    IMPORTANT - GESTION DU CONTEXTE:
- Si l'utilisateur pose une question de SUIVI (ex: "oui", "dis-moi plus", "et concrètement ?"), 
  TU DOIS utiliser les informations de la CONVERSATION PRÉCÉDENTE.
- Ne redemande PAS les informations déjà fournies.
- Réutilise les données du dernier échange si elles sont toujours pertinentes.

INFORMATIONS À EXTRAIRE:

1. **depense_totale** (float, obligatoire): 
   - La dépense du MOIS EN COURS (mois actuel, ce mois)
   - Cherche: "ce mois", "j'ai dépensé X euros", "dépense actuelle"
   - Si non mentionné ET qu'il y a une conversation précédente: RÉUTILISE la valeur précédente
   - Sinon: estimer à depense_prev * 1.05

2. **depense_prev** (float, obligatoire):
   - La dépense du MOIS PRÉCÉDENT (mois passé, dernier mois)
   - Cherche: "le mois passé", "le mois dernier", "j'avais dépensé"
   - Si non mentionné ET qu'il y a une conversation précédente: RÉUTILISE la valeur précédente
   - Sinon: estimer à depense_totale * 0.95

3. **salaire** (float, obligatoire):
   - Le salaire mensuel NET de l'utilisateur
   - Cherche: "je gagne", "mon salaire est", "mes revenus"
   - Si non mentionné ET qu'il y a une conversation précédente: RÉUTILISE la valeur précédente
   - Valeur minimale: 0

4. **loyer** (float, obligatoire):
   - Le montant du loyer mensuel
   - Cherche: "mon loyer", "je paie X de loyer", "loyer de"
   - Si non mentionné ET qu'il y a une conversation précédente: RÉUTILISE la valeur précédente
   - Sinon: estimer automatiquement à 30% du salaire

5. **nb_enfants** (int, optionnel):
   - Nombre d'enfants à charge
   - Si non mentionné ET qu'il y a une conversation précédente: RÉUTILISE la valeur précédente
   - Par défaut: 0

6. **lag_1**, **lag_2**, **lag_3** (float, optionnels):
   - Dépenses des mois précédents
   - Laisser à null si non mentionnées

RÈGLES D'EXTRACTION IMPORTANTES:

 **Questions de suivi**: 
   - "oui", "ok", "dis-moi plus", "et concrètement ?", "peux-tu détailler ?"
   -> Ces messages ne contiennent PAS de nouvelles données financières
   -> RÉUTILISE TOUTES les données de la conversation précédente

 **Distinction temporelle**: 
   - "ce mois" / "en ce moment" -> depense_totale
   - "le mois passé" / "le mois dernier" -> depense_prev

 **Contexte conversationnel**: 
   - Si l'historique contient des informations et que le message actuel n'en contient pas,
     c'est probablement une question de suivi -> garde les mêmes données

 **Format de sortie**: Retourne UNIQUEMENT le JSON, AUCUN texte avant ou après

EXEMPLES D'EXTRACTION:

Exemple 1 - Premier message:
Message: "Je suis étudiant et je gagne 615 euros par mois. Le mois passé j'ai dépensé 150 euros et ce mois j'ai dépensé 110 euros. Mon loyer est de 299.5 euros."

Extraction:
{{{{
    "depense_totale": 110.0,
    "depense_prev": 150.0,
    "salaire": 615.0,
    "loyer": 299.5,
    "nb_enfants": 0,
    "lag_1": null,
    "lag_2": null,
    "lag_3": null
}}}}

Exemple 2 - Question de suivi (l'utilisateur avait déjà donné ses infos):
Historique: L'utilisateur avait dit "Je gagne 2000€, j'ai dépensé 1000€ ce mois, loyer 600€"
Message actuel: "oui tu me conseil quoi concrètement"

Extraction (RÉUTILISE les données précédentes):
{{{{
    "depense_totale": 1000.0,
    "depense_prev": 950.0,
    "salaire": 2000.0,
    "loyer": 600.0,
    "nb_enfants": 0,
    "lag_1": null,
    "lag_2": null,
    "lag_3": null
}}}}

Exemple 3 - Mise à jour partielle:
Historique: L'utilisateur avait donné salaire=2500€, loyer=700€
Message actuel: "ce mois j'ai dépensé 1300€"

Extraction (garde salaire et loyer, ajoute la nouvelle dépense):
{{{{
    "depense_totale": 1300.0,
    "depense_prev": 1235.0,
    "salaire": 2500.0,
    "loyer": 700.0,
    "nb_enfants": 0,
    "lag_1": null,
    "lag_2": null,
    "lag_3": null
}}}}

FORMAT DE SORTIE ATTENDU:
{format_instructions}

RAPPEL CRITIQUE: 
- Retourne UNIQUEMENT le JSON valide, sans texte additionnel
- En cas de question de suivi, RÉUTILISE les données de l'historique
- Ne laisse JAMAIS le salaire à 0 si l'utilisateur l'a déjà mentionné dans l'historique"""
    
    def extract_financial_info(self, user_message: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Extrait les informations financières du message utilisateur 
        en tenant compte de l'historique du chat,
        avec gestion d'erreur robuste et logging
        """
        
        if chat_history is None:
            chat_history = []
        
        try:
            # Formatage de l'historique pour LangChain
            formatted_history = self._format_history_for_llm(chat_history)
        
            # Appel du LLM pour extraction
            chain = self.extraction_prompt | self.llm | self.parser
            result = chain.invoke({
                "user_message": user_message,
                "chat_history": formatted_history
            })
        
            # Conversion en dict
            extracted_data = result.dict()
        
            # Validation des données critiques
            if extracted_data.get('salaire', 0) <= 0:
                print("Warning: Salaire <= 0 détecté, tentative de récupération depuis l'historique")
                # Chercher les infos dans l'historique
                last_valid_data = self._extract_from_history(chat_history)
                if last_valid_data:
                    print("Données récupérées depuis l'historique")
                    return last_valid_data
                else:
                    print("Pas de données dans l'historique, utilisation du fallback")
                    return self._get_fallback_data()
        
            # Log pour debug
            print(f"Extraction réussie: salaire={extracted_data.get('salaire')}€, "
                f"depense_totale={extracted_data.get('depense_totale')}€")
        
            return extracted_data
        
        except json.JSONDecodeError as e:
            print(f"Erreur JSON: {str(e)}")
            # Tenter de récupérer depuis l'historique
            last_valid_data = self._extract_from_history(chat_history)
            return last_valid_data if last_valid_data else self._get_fallback_data()
    
        except Exception as e:
            print(f"Erreur d'extraction: {type(e).__name__} - {str(e)}")
            # Tenter de récupérer depuis l'historique
            last_valid_data = self._extract_from_history(chat_history)
            return last_valid_data if last_valid_data else self._get_fallback_data()
        
    def _extract_from_history(self, chat_history: List[Dict]) -> Optional[Dict[str, Any]]:
        """
        Cherche les dernières données financières valides dans l'historique
        """
        for message in reversed(chat_history):
            if message['role'] == 'assistant' and 'prediction_data' in message:
                # Récupérer les features utilisées pour la dernière prédiction
                features = message['prediction_data'].get('features_utilisees', {})
                if features and features.get('salaire', 0) > 0:
                    return {
                        'depense_totale': features.get('depense_totale', 1000),
                        'depense_prev': features.get('depense_prev', 950),
                        'salaire': features.get('salaire', 2000),
                        'loyer': features.get('loyer', 600),
                        'nb_enfants': features.get('nb_enfants', 0),
                        'lag_1': features.get('lag_1'),
                        'lag_2': features.get('lag_2'),
                        'lag_3': features.get('lag_3')
                    }
        return None
        
    def _get_fallback_data(self) -> Dict[str, Any]:
        """Données par défaut en cas d'échec d'extraction"""
        return {
            'depense_totale': 1000.0,
            'depense_prev': 950.0,
            'salaire': 2000.0,
            'loyer': 600.0,
            'nb_enfants': 0,
            'lag_1': None,
            'lag_2': None,
            'lag_3': None
        }
    
    # FORMATAGE DE L'HISTORIQUE POUR LE LLM
    def _format_history_for_llm(self, chat_history: List[Dict]) -> List:
        """Convertit l'historique de Streamlit en format LangChain (HumanMessage/AIMessage)"""
        formatted_messages = []
        for message in chat_history:
            if message['role'] == 'user':
                formatted_messages.append(HumanMessage(content=message['content']))
            elif message['role'] == 'assistant':
                formatted_messages.append(AIMessage(content=message['content']))
        return formatted_messages
    
    def generate_conversational_response(
        self, 
        user_message: str,
        extracted_info: Dict,
        prediction_result: Dict,
        chat_history: List[Dict]
    ) -> str:
        
        conseils_text = "\n".join([f"• {c}" for c in prediction_result['conseils']])
        ci_low, ci_high = prediction_result['confidence_interval']
        confidence_text = f"entre {ci_low}€ et {ci_high}€"
        
        # Créer la chaîne
        chain = self.response_prompt | self.llm
        
        # Générer la réponse
        response = chain.invoke({
            "user_message": user_message,
            "extracted_info": json.dumps(extracted_info, indent=2, ensure_ascii=False),
            "prediction": prediction_result['prediction'],
            "confidence": confidence_text,
            "profil": f"{prediction_result['profil_detecte']} - Niveau {prediction_result['tier_conso']}",
            "conseils": conseils_text,
            "chat_history": self._format_history_for_llm(chat_history)
        })
        
        return response.content
    
    def process_user_query(self, user_message: str, predictor, chat_history: List[Dict]) -> Dict[str, Any]:
        """
        Pipeline complet: Extraction - Prédiction - Réponse
        """
        
        extracted_info = self.extract_financial_info(user_message, chat_history)
        
        prediction = predictor.predict(extracted_info)
        
        response = self.generate_conversational_response(
            user_message, 
            extracted_info, 
            prediction,
            chat_history
        )
        
        return {
            'extracted_info': extracted_info,
            'prediction': prediction,
            'response': response
        }

