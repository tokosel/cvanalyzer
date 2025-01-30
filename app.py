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
    """Analyse du CV avec perspective de recrutement"""
    analysis_prompts = {
        "matching": f"""Agis en chargé de recrutement expert. 
        Analyse ce CV par rapport à ce poste :
        Descriptif du poste : {job_description}
        CV analysé : {cv_text}
        Évalue précisément :
        - Correspondance globale du profil
        - Compétences clés
        - Adéquation formation/expérience
        - Potentiel pour le poste
        Format :
        🎯 Taux de Matching : X%
        🔑 Compétences Clés Alignées : [liste]
        🚨 Points à Améliorer : [liste]
        💡 Recommandations : [conseils]""",
        
        "technical": f"""Analyse technique du CV :
        Poste : {job_description}
        CV : {cv_text}
        Filtres :
        - Technologies requises
        - Niveau technique
        - Certifications
        - Expériences techniques précises
        Rapport :
        ✅ Technologies Maîtrisées : [liste]
        ❌ Technologies Manquantes : [liste]
        📊 Score Technique : X/10""",
        
        "psychological": f"""Analyse comportementale du candidat :
        Contexte : {job_description}
        CV : {cv_text}
        Évaluation :
        - Soft skills
        - Adaptabilité
        - Potentiel de développement
        - Alignement culturel
        Insights :
        🧠 Profil Psychologique : [description]
        🤝 Compatibilité Culturelle : X%
        🚀 Potentiel de Croissance : [évaluation]"""
    }
    
    prompt = analysis_prompts.get(analysis_type, analysis_prompts["matching"])
    model = genai.GenerativeModel("gemini-1.5-flash-8b-exp-0827")
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