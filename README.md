# GPU-projects avec C++ et CUDA

## Description du projet

Ce projet exploite les capacités de calcul des cartes graphiques NVIDIA pour réaliser des calculs parallèles en C++ avec CUDA.  
L'objectif est de comparer les performances entre CPU et GPU, en mesurant les images par seconde (FPS) pour différentes applications.

Grâce à ce projet, j'ai appris à utiliser les fonctions GPU pour paralléliser les calculs et résoudre des problèmes algorithmiques et techniques nécessitant une forte puissance de calcul. Cela donne un rendu très visuel et on comprend l'enjeu de l'optimisation de ces performances.

## Applications réalisées

1. **Fractales – Ensemble de Julia** 
   Visualisation mathématique des fractales, calculée et rendue en parallèle sur GPU et CPU.

2. **Ray Tracing** 
   Rendu 3D avec effets lumineux réalistes, exploitant le parallélisme pour accélérer le calcul des rayons et des ombres.

3. **Jeu de la vie (Game of Life)** 
   Simulation de systèmes complexes avec règles locales appliquées en parallèle sur une grille de cellules.

4. **Simulation dynamique de nombreux corps/points** 
   Calculs physiques et interactions entre de nombreux objets (N-Body), visualisation en temps réel avec OpenGL.

5. **K-means** 
   Clustering et traitement de grandes quantités de données, avec assignation des points aux clusters sur CPU et GPU pour comparaison de performance.

## Objectifs pédagogiques

- Maîtriser les calculs parallèles sur GPU avec CUDA.  
- Comprendre les différences de performances entre CPU et GPU.  
- Appliquer les calculs parallèles à la génération d’objets 3D, au rendu réaliste et à la simulation de systèmes complexes. ✨  
- Développer des compétences en visualisation graphique avec OpenGL.

## Résultats

Les travaux réalisés permettent de constater une **amélioration significative des performances** sur GPU par rapport au CPU pour tous les calculs massivement parallèles (fractales, ray tracing, K-means, N-Body, Game of Life).  
Les visualisations en temps réel démontrent l’efficacité du parallélisme GPU pour le rendu et la simulation dynamique.

## Technologies utilisées

- C++ et CUDA pour le calcul parallèle.  
- OpenGL/GLUT pour la visualisation graphique.  
- NVIDIA GPU pour l’accélération des calculs.  

## Prérequis

- **Visual Studio 2022** (le projet a été initialement développé sous VS2019 voir avant, mais fonctionne sous VS2022).  
- **CUDA Toolkit** (dans ce projet 12.9) et donc un appareil compatible (développements faits avec une GTX 1060).
- **Windows 10 ou 11**.  
- Bibliothèques incluses dans le dépôt :  
  - **FreeGLUT**  
  - **GLM**  
  - **GLEW** (si nécessaire)  
- Les fichiers DLL nécessaires doivent être placés dans **C:\Windows\System32** ou ajoutés au **PATH**.

## Installation

1. **Cloner le dépôt**  
   ```bash
   git clone <URL_DU_DEPOT>
   ```

2. **Ouvrir la solution dans Visual Studio**  
   - Double-cliquez sur `GPU.sln` pour ouvrir la solution dans Visual Studio 2022.

3. **Vérifier les dépendances**  
   - Les bibliothèques FreeGLUT, GLM et GLEW sont incluses dans le dossier `External Libraries/`, à extraire dans ce dossier.  
   - Dans Visual Studio, faites un clic droit sur le projet → **Propriétés** → **VC++ Directories** :  
     - **Include Directories** : pour chaque lib, ajoutez le chemin vers `External Libraries/include`.  
     - **Library Directories** : pour chaque lib, ajoutez le chemin vers `External Libraries/lib`.  
   - Dans **Linker → Input → Additional Dependencies**, ajoutez les fichiers `.lib` nécessaires dans la bonne configuration :  
     - `freeglut.lib`  
     - `glew32.lib`

4. **CUDA**  
   - Télécharger et installer le [CUDA Toolkit officiel](https://developer.nvidia.com/cuda-toolkit-archive).   
   - Vérifier que le driver du GPU est à jour avant de lancer l'installer.
   - Une fois installé, vérifier que les projets sont bien liés à la version du toolkit installé:
     - **Répertoires VC ++** : pour chaque projet, vérifier ou ajouter les includes dans **Répertoires Include** : `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\{version}\include\`.
     - **.vcxproj** : pour chaque ptojet, ouvrir dans un éditeur et vérifier/modifier le numéro de version:
         - <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA {version}.props" />
         - <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA {version}.targets" />

5. **DLL**  
   - Copiez les fichiers `*.dll` requis dans :  
     - `C:\Windows\System32`  
     - **ou** dans le dossier contenant l’exécutable généré (`Debug/` ou `Release/`).  
   - Par exemple : `freeglut.dll`, `glew32.dll`.

6. **Compiler et exécuter**  
   - Sélectionnez **Debug** ou **Release** et compilez le projet.  
   - Exécutez les fichiers .exe de chaque projet depuis Visual Studio ou dans le dossier `Debug/` ou `Release/`.

## Contrôles / Inputs

### Julia Set
- **Flèches directionnelles** : déplacer la vue dans la fractale  
- **+ / -** : zoomer / dézoomer  
- **S** : sauvegarder l’image rendue  
- **Esc** : quitter  

### Ray Tracing
- **Souris** : rotation de la caméra  
- **Flèches** : déplacer la caméra  
- **1 / 2** : basculer entre mode CPU et GPU  
- **Esc** : quitter  

### K-means
- **1 / 2** : basculer entre mode CPU et GPU  
- **Flèche Haut** : doubler le nombre de points  
- **Flèche Bas** : réduire de moitié le nombre de points  
- **Esc** : quitter  

### Jeu de la vie
- **Espace** : démarrer / arrêter la simulation  
- **Flèches directionnelles** : déplacer la grille  
- **+ / -** : augmenter ou diminuer la vitesse d’évolution  
- **Esc** : quitter  

### N-body / Many Points
- **Flèche Haut** : augmenter le nombre de corps simulés  
- **Flèche Bas** : diminuer le nombre de corps  
- **1 / 2 / 3 / 4** : changer le mode d’exécution (CPU, GPU, GPU double buffer, GPU mémoire partagée)  
- **Esc** : quitter 

## Notes

- Les performances FPS sont affichées dans le titre de la fenêtre.
- Tous les projets permettent une comparaison CPU vs GPU pour observer les gains de parallélisation.
- Bon moyen d’apprendre la programmation parallèle et l’optimisation GPU pour des simulations et rendus complexes.

## License

- Projet réalisé dans le cadre de travaux pratiques sur le calcul parallèle et la programmation GPU. – usage libre pour consultation, tests et apprentissage.
