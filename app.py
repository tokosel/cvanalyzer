# app.py
from flask import Flask, render_template, request, jsonify
import os
from dotenv import load_dotenv
import google.generativeai as genai
import fitz  # PyMuPDF
import tempfile
import logging


app = Flask(__name__)
load_dotenv()

# Configuration de l'API Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def pdf_to_text(pdf_file):
    """Extraire le texte de toutes les pages du PDF"""
    text = ""
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f'temp_pdf_{os.getpid()}_{id(pdf_file)}.pdf')
    
    try:
        # Sauvegarder le fichier temporairement
        pdf_file.save(temp_path)
        
        # Ouvrir et lire le PDF
        with fitz.open(temp_path) as pdf_document:
            for page in pdf_document:
                text += page.get_text()
                
    except Exception as e:
        logging.error(f"Erreur lors de la lecture du PDF: {str(e)}")
        raise
    
    finally:
        # Nettoyer le fichier temporaire
        try:
            if os.path.exists(temp_path):
                pdf_document.close()  # S'assurer que le document est fermé
                os.remove(temp_path)
        except Exception as e:
            logging.error(f"Erreur lors de la suppression du fichier temporaire: {str(e)}")
    
    return text

def get_gemini_response(job_description, cv_text, analysis_type):
    """Analyse du CV avec perspective de coaching pour chercheurs d'emploi"""
    
    analysis_prompts = {
        "matching": f"""En tant que coach en recherche d'emploi, je vais vous aider à optimiser votre candidature pour ce poste.
        
        📌 **Descriptif du poste** : {job_description}
        📄 **Votre CV** : {cv_text}
        
        **Analyse & Conseils Personnalisés :**
        - 🔍 Correspondance de votre profil avec l’offre
        - ✅ Vos points forts à valoriser
        - 🔧 Axes d’amélioration
        - 🎯 Stratégies concrètes pour maximiser vos chances d’être recruté(e)
        
        **Format du retour :**
        🎯 **Niveau d’adéquation au poste** : X%  
        💡 **Forces et atouts à mettre en avant** : [liste]  
        🚀 **Points à améliorer & recommandations** : [liste avec actions précises]  
        📌 **Conseils pour renforcer votre candidature** : [actions pratiques]""",
        
        "technical": f"""Je vais analyser votre profil technique et vous donner des conseils pour optimiser votre positionnement sur le marché du travail.
        
        📌 **Poste ciblé** : {job_description}
        📄 **Votre CV** : {cv_text}
        
        **Analyse et recommandations :**
        - ✅ Technologies et outils que vous maîtrisez
        - ❌ Compétences techniques à renforcer
        - 🎓 Formations et certifications à valoriser ou à acquérir
        - 📊 Stratégies pour améliorer votre attractivité auprès des recruteurs
        
        **Format du retour :**
        ✅ **Compétences techniques mises en avant** : [liste]  
        ❌ **Manques identifiés et suggestions d’amélioration** : [liste]  
        📌 **Formations ou certifications à envisager** : [recommandations]  
        🚀 **Conseils pour renforcer votre profil technique** : [actions pratiques]""",
        
        "psychological": f"""Je vais analyser votre profil comportemental afin de vous donner des conseils pour mieux vous positionner et réussir vos entretiens.
        
        📌 **Contexte du poste** : {job_description}
        📄 **Votre CV** : {cv_text}
        
        **Évaluation des soft skills & coaching personnalisé :**
        - 🧠 Votre profil psychologique
        - 🤝 Votre capacité à vous intégrer dans une équipe / culture d'entreprise
        - 🚀 Votre potentiel d’évolution et vos axes de développement
        - 🔥 Conseils pour valoriser vos soft skills en entretien et sur votre CV
        
        **Format du retour :**
        🧠 **Vos forces comportementales** : [description]  
        🤝 **Votre niveau d’alignement avec la culture d’entreprise** : X%  
        🚀 **Axes d’amélioration et conseils de développement personnel** : [liste]  
        🎤 **Stratégies pour mieux vous vendre en entretien** : [conseils]"""
    }
    
    prompt = analysis_prompts.get(analysis_type, analysis_prompts["matching"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    
    return response.text


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        job_description = request.form.get('job_description')
        analysis_type = request.form.get('analysis_type')
        files = request.files.getlist('cv_files')
        
        results = []
        for file in files:
            if file.filename.endswith('.pdf'):
                try:
                    cv_text = pdf_to_text(file)
                    analysis = get_gemini_response(job_description, cv_text, analysis_type)
                    results.append({
                        'filename': file.filename,
                        'analysis': analysis
                    })
                except Exception as e:
                    logging.error(f"Erreur lors de l'analyse du fichier {file.filename}: {str(e)}")
                    return jsonify({
                        'success': False,
                        'error': f"Erreur lors de l'analyse du fichier {file.filename}: {str(e)}"
                    })
        
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        logging.error(f"Erreur générale: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)