# 🍌 Gemini Nano Banana for ComfyUI

Ce projet est un custom node pour [ComfyUI](https://github.com/comfyanonymous/ComfyUI) qui intègre la puissance de l'API **Google Gemini 2.5 Flash Image ("Nano Banana")**.  
Il fournit un node unique et polyvalent, le **Gemini Nano Studio**, qui permet de réaliser des opérations de génération et d'édition d'images directement dans vos workflows.

---

## ✨ Caractéristiques

- **Node Universel** : Un seul node pour gouverner toutes les tâches !
  - **Text-to-Image** : Générez des images à partir d'un simple prompt texte.
  - **Image-to-Image** : Transformez une image existante en suivant vos instructions.
  - **Fusion Multi-Image** : Envoyez jusqu'à 4 images pour des opérations de fusion, de mélange de styles ou de composition complexes.
- **Sécurité de la Clé API** : Votre clé API est lue depuis un fichier local `apikey.txt` et n'est **jamais** sauvegardée dans les métadonnées de vos images ou les fichiers de workflow, garantissant un partage sécurisé.
- **Upscaler Intégré** : Améliorez la qualité de vos générations directement à la sortie du node en utilisant vos modèles d'upscale ComfyUI préférés (ESRGAN, etc.).
- **Sortie Texte** : En plus de l'image, le node fournit une sortie texte pour afficher les réponses ou les messages du modèle Gemini, facilitant le débogage et le prompt engineering.

---

## 🚀 Installation

### 1. Prérequis

Assurez-vous d'avoir ComfyUI à jour. Ce node nécessite la librairie Python de Google.

Ouvrez un terminal, naviguez jusqu'à votre dossier **ComfyUI**, activez votre environnement virtuel (`venv`) et lancez la commande suivante :

```bash
pip install google-generativeai opencv-python
```

---

### 2. Installation du Node

**Méthode avec Git :**  
Naviguez dans votre dossier `ComfyUI/custom_nodes/` et clonez ce dépôt :

```bash
git clone https://github.com/orion4d/Gemini_Banana_by_orion4d.git
```

**Méthode Manuelle :**  
Téléchargez ce projet en tant que fichier ZIP et extrayez-le dans votre dossier `ComfyUI/custom_nodes/`.

---

### 3. Configuration de la Clé API (Très Important !)

Ce node utilise un fichier local pour stocker votre clé API de manière sécurisée.

1. Allez dans le dossier du node : `ComfyUI/custom_nodes/ComfyUI_Gemini_Nano/`.
2. Créez un nouveau fichier texte nommé `apikey.txt`.
3. Ouvrez ce fichier et collez-y **votre clé API Google**, et rien d'autre.
4. Sauvegardez le fichier.

---

### 4. Redémarrage

Redémarrez **ComfyUI** pour que le nouveau node soit chargé.

---

## 🎨 Utilisation

Une fois installé, vous trouverez le node **« 🍌 Gemini Nano Studio »** dans la catégorie **« Gemini Banana »**.

### Mode Text-to-Image

Pour générer une image à partir de texte, **ne connectez aucune image aux entrées**.  
Écrivez simplement votre prompt.

*(Pensez à remplacer ce lien par votre capture d'écran de T2I.)*

---

### Mode Image-to-Image / Fusion

Connectez une ou plusieurs images pour les éditer ou les combiner.

- **Édition simple** : Connectez une image à `image1` et décrivez la modification dans le prompt.
- **Fusion** : Connectez plusieurs images et décrivez comment les fusionner.

*(Pensez à remplacer ce lien par votre capture d'écran d'édition.)*

---

## 💡 L'Art du Prompt Engineering

Pour obtenir les meilleurs résultats, soyez directif.  
Le modèle Gemini peut parfois répondre avec du texte au lieu d'une image.  
Pour **forcer la génération d'image**, commencez votre prompt par une instruction claire :

**Exemple de prompt efficace :**

```text
Return ONLY the modified image. Do not add any text.
In the input image, replace the text "ENCRYPTMASTER" with "ORION4D".
The new text should match the original's neon blue, stylized, and energetic typography.
```

---

## 🔧 Dépannage

- **Erreur 429 (Quota Exceeded)** : Vous avez fait trop de requêtes à l'API trop rapidement. Attendez 2-3 minutes avant de relancer une génération.
- **Node non trouvé** : Assurez-vous d'avoir bien redémarré ComfyUI après l'installation.
- **Erreur « Fichier `apikey.txt` non trouvé »** : Vérifiez que le fichier `apikey.txt` est bien présent à la racine du dossier du node et qu'il n'est pas vide.

---
<div align="center">

<h3>🌟 <strong>Show Your Support</strong></h3>
<p>If this project helped you, please consider giving it a ⭐ on GitHub!</p>
<p><strong>Made with ❤️ for the ComfyUI community</strong></p>
<p><strong>by Orion4D</strong></p>
<a href="https://ko-fi.com/orion4d">
<img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Buy Me A Coffee" height="41" width="174">
</a>

</div>
