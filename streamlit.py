import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from imblearn.over_sampling import RandomOverSampler
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("complete.csv")

df = load_data()

# Menu de navigation
menu = st.sidebar.radio(
    "Navigation",
    ("Accueil", "Exploration data", "Data Visualization", "Modélisation","Conclusion"))

# Fonction pour afficher la page d'accueil
def page_accueil():
    st.title("Projet de classification des feux de forêt aux USA")
    st.image("image feu.png")
    st.subheader("Objectif de l'étude")
    st.write("Le National Wildfire Coordinating Group (NWCG) a été créé aux États-Unis à la suite des conséquences d'une importante saison d'incendies de forêt en 1970, notamment l' incendie de Laguna.")
    st.write("Ce groupe gère plusieurs publications contenant une base de données spatiales sur les incendies de forêt survenus aux États-Unis de 1992 à 2015.")
    st.write("L’objectif de cette collecte de données est de pouvoir étudier l’origine des feux et de pouvoir anticiper les prochains départs de feu et de mieux les gérer afin de limiter les impacts humains et économiques aux Etats-Unis.")

# Fonction pour afficher la page 1
def page_1():
    st.title("Exploration data")
    st.subheader("Dataframe des feux de forêt aux USA de 1992 à 2015 et des températures moyennes de 1950 à 2022")
    st.write("Pour réaliser ces analyses prédictives j'ai utilisé 2 dataframes disponibles sur le site web Kaggle:")
    st.write("- Lien Kaggle sur les données des feux USA: https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires?resource=download")
    st.write("- Lien Kaggle sur les données des températures moyennes USA: https://www.kaggle.com/datasets/justinrwong/average-monthly-temperature-by-us-state")
    st.write("Dataframe contenant le merge de 2 dataframes des feux de forêt et des températures moyennes par états:")
    st.dataframe(df.head(10))
    st.subheader("Traitement des données")
    st.write("Suppression des colonnes inutiles du type : FOD_ID, FPA_ID, SOURCE_SYSTEM_TYPE, … car celles-ci n'apportent pas d’informations pertinentes dans l’analyse des données.")
    st.write("Conversion au format datetime.")
    st.write("Gestion des données manquantes via par exemple la création d'une nouvelle variable comme date_diff afin de copier la date de la découverte de feu dans la variable de déclaration de feu qui contient des valeurs manquantes.")
    st.write("Merge des deux jeux de données après traitement. Pour cela  il a fallu créer un dictionnaire state_mapping de mapping afin de pouvoir fusionner les 2 dataframes.")
    st.write("J'ai aussi réalisé un encodage cyclique des dates via les fonctions sinus et cosinus afin que nos dates soient en format numérique pour pouvoir réaliser nos modélisations.")

# Fonction pour afficher la page 2
def page_2():
    st.title("Data Visualization")
    st.write("Ci-dessous les différentes datavisualisation des feux de forêt afin de pouvoir identifier les paramètres qui permettront de définir notre modèle prédictif:")

     # Options disponibles pour les graphiques
    viz_options = [
        "Nombre de feux par année et par classe",
        "Nombre de feux par mois et par classe",
        "Cartographie des classes de feu aux USA",
        "Zoom sur les classes de feu en Californie"
    ]

    # Multiselect pour permettre une sélection dynamique
    selected_viz = st.multiselect("Sélectionnez les graphiques à afficher :", options=viz_options)

    # Affichage conditionnel des graphiques
    if "Nombre de feux par année et par classe" in selected_viz:
        st.subheader("Nombre de feux par année et par classe:")
        fig = plt.figure(figsize=(15, 6))
        sns.countplot(x='FIRE_YEAR', hue='FIRE_SIZE_CLASS', data=df)
        st.pyplot(fig)
        st.write("La représentation graphique ci-dessous démontre que pour certaines années le nombre de feux est supérieur aux autres. Pour le moment difficile de déterminer réellement quels paramètres fait que ce nombre évolue.")


    if "Nombre de feux par mois et par classe" in selected_viz:
        st.subheader("Nombre de feux par mois et par classe:")
        fig = plt.figure(figsize=(12, 6))
        sns.countplot(x='CONT_MONTH', hue='FIRE_SIZE_CLASS', data=df)
        st.pyplot(fig)
        st.write("La représentation graphique ci-dessous démontre que le nombre de feux est très nombreux pendant la période d’été de Juin à Septembre quel que soit la classe de feu. Les feux de type A et B sont très représentés pendant la période estivale américaine avec un facteur de plus de 4 entre Janvier et Août.")

    if "Cartographie des classes de feu aux USA" in selected_viz:
        st.subheader("Cartographie des classes de feu aux USA:")
        st.image("Categorie feu aux USA.png")
        st.write("Cette représentation permet de démontrer que les régions de l’ouest américain ont un nombre de feux plus élevés que les régions de l’est et que les feux de classe E.")

    if "Zoom sur les classes de feu en Californie" in selected_viz:
        st.subheader("Zoom sur les classes de feu en Californie:")
        st.image("Feux Californie.png")
        st.write("Sur cette modélisation on peut constater que les feux qui font partie des classes les plus importantes sont surtout localisés au sud de la Californie.")

# Fonction pour afficher la page 3
def page_3():
    st.title("Modélisation")
    st.write("Analyse prédictive basée sur les données.")

    st.header("Modelisation",divider=True)
    st.subheader("Modélisation régression logistique")
    st.write("J'ai tout d'abord réalisé une modélisation par régression logistique. L'utilisation de ce modèle à pour objectif de vérifier s' il existe une relation linéaire entre les classes de feu, la localisation par état et les moyennes de températures annuelles. Les résultats sont peu satisfaisants démontrant que ce modèle prédictif n’est pas le mieux adapté. C’est pour cela que j’ai choisi d’utiliser des modèles de random forest ou XGBClassifier.")

    st.subheader("Modélisation random forest et XGBClassifier")
    st.write("Pour chacun des modèles j'ai utilisé une méthode d'échantillonnage random over sampling qui consiste à compléter les données de formation par des copies multiples de certaines d'instances de la classe minoritaire. ")
    st.write("Les résultats obtenus sont nettement supérieurs au modèle de régression logistique et démontre que ces modèles prédictifs sont plus adaptés pour la définition des classes de feu.")
    st.write("Afin d’améliorer les résultats nous avons décidé de regrouper les classes de feu afin de rationaliser les classes de feu:")
    st.write("- Les classes A et B dans la classe A")
    st.write("- Les classes C et D dans la classe B")
    st.write("- Les classes E dans la classe C")
    st.write("- Les classes F et G dans la classe D")
    st.write("J'ai également une feature importance sur les 2 modèles afin d'identifier les 4 variables qui ont le plus d'impact sur la modélisation:")
    st.write("- Feature importance random forest")
    st.image("feature importance random.png")
    st.write("- Feature importance XGClassifier")
    st.image("feature importance XGClassifier.png")

    Weather_fires = df[['FIRE_YEAR','DISCOVERY_DAY','DISCOVERY_MONTH','DISCOVERY_YEAR','CONT_DAY','CONT_MONTH','CONT_YEAR','STAT_CAUSE_DESCR','FIRE_SIZE','FIRE_SIZE_CLASS','LATITUDE','LONGITUDE','STATE','average_temp','monthly_mean_from_1901_to_2000']]
    Weather_fires = Weather_fires.sort_values(by=['FIRE_YEAR','DISCOVERY_MONTH'])

    # Define the periods for the cyclic variables
    period_day = 31
    period_month = 12

    # Apply cyclic encoding to DISCOVERY_DAY and CONT_DAY
    Weather_fires['DISCOVERY_DAY_sin'] = np.sin(2 * np.pi * Weather_fires['DISCOVERY_DAY'] / period_day)
    Weather_fires['DISCOVERY_DAY_cos'] = np.cos(2 * np.pi * Weather_fires['DISCOVERY_DAY'] / period_day)
    Weather_fires['CONT_DAY_sin'] = np.sin(2 * np.pi * Weather_fires['CONT_DAY'] / period_day)
    Weather_fires['CONT_DAY_cos'] = np.cos(2 * np.pi * Weather_fires['CONT_DAY'] / period_day)

    # Apply cyclic encoding to DISCOVERY_MONTH and CONT_MONTH
    Weather_fires['DISCOVERY_MONTH_sin'] = np.sin(2 * np.pi * Weather_fires['DISCOVERY_MONTH'] / period_month)
    Weather_fires['DISCOVERY_MONTH_cos'] = np.cos(2 * np.pi * Weather_fires['DISCOVERY_MONTH'] / period_month)
    Weather_fires['CONT_MONTH_sin'] = np.sin(2 * np.pi * Weather_fires['CONT_MONTH'] / period_month)
    Weather_fires['CONT_MONTH_cos'] = np.cos(2 * np.pi * Weather_fires['CONT_MONTH'] / period_month)

    # Assume a cyclical year period of 10 years for this example
    period_year = 10

    # Apply cyclic encoding to DISCOVERY_YEAR and CONT_YEAR
    Weather_fires['DISCOVERY_YEAR_sin'] = np.sin(2 * np.pi * (Weather_fires['DISCOVERY_YEAR'] % period_year) / period_year)
    Weather_fires['DISCOVERY_YEAR_cos'] = np.cos(2 * np.pi * (Weather_fires['DISCOVERY_YEAR'] % period_year) / period_year)
    Weather_fires['CONT_YEAR_sin'] = np.sin(2 * np.pi * (Weather_fires['CONT_YEAR'] % period_year) / period_year)
    Weather_fires['CONT_YEAR_cos'] = np.cos(2 * np.pi * (Weather_fires['CONT_YEAR'] % period_year) / period_year)

    Weather_fires= Weather_fires[['FIRE_SIZE_CLASS','LATITUDE','LONGITUDE','average_temp','monthly_mean_from_1901_to_2000','DISCOVERY_DAY_sin', 'DISCOVERY_DAY_cos', 'DISCOVERY_MONTH_sin', 'DISCOVERY_MONTH_cos',
        'DISCOVERY_YEAR_sin', 'DISCOVERY_YEAR_cos', 'CONT_DAY_sin', 'CONT_DAY_cos',
        'CONT_MONTH_sin', 'CONT_MONTH_cos', 'CONT_YEAR_sin', 'CONT_YEAR_cos']]

  
    #séparation des classes de feu
    Weather_fires['FIRE_SIZE_CLASS']= Weather_fires['FIRE_SIZE_CLASS'].replace(['A','B'],['A','A'])
    Weather_fires['FIRE_SIZE_CLASS']= Weather_fires['FIRE_SIZE_CLASS'].replace(['C','D'],['B','B'])
    Weather_fires['FIRE_SIZE_CLASS']= Weather_fires['FIRE_SIZE_CLASS'].replace(['E'],['C'])
    Weather_fires['FIRE_SIZE_CLASS']= Weather_fires['FIRE_SIZE_CLASS'].replace(['F','G'],['D','D'])

    #Séparation du jeu de données
    #Séparer les données en un DataFrame feats contenant les variables explicatives et un DataFrame target contenant la variable cible classe des feux
    feats = Weather_fires.drop('FIRE_SIZE_CLASS', axis=1)
    target = Weather_fires['FIRE_SIZE_CLASS']

    #Séparer le jeu de données en un jeu d'entraînement (X_train,y_train) et un jeu de test (X_test, y_test) de sorte que la partie de test contient 25% du jeu de données initial.
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state = 42, shuffle = False)

    #Encodage les modalités de la variable cible FIRE_SIZE_CLASS à l'aide d'un LabelEncoder en estimant l'encodage sur le jeu d'entraînement et en l'appliquant sur le jeu d'entraînement et de test
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()

    y_train = le.fit_transform(y_train)

    y_test = le.transform(y_test)

    #Echantillonnage des données
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X_train, y_train)

# Fonction permettant de choisir le modèle
    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = ensemble.RandomForestClassifier(n_jobs=-1, random_state=321)
        elif classifier == 'XGBClassifier':  # Correction du nom ici
            clf = XGBClassifier(n_estimators=50, max_depth=50)
        else:
            st.error("Modèle non reconnu. Veuillez vérifier votre sélection.")
            return None
        clf.fit(X_train, y_train)
        return clf

 # Fonction permettant de choisir la restitution des résultats pour chacun des modèles utilisés
    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_res, y_res)
        elif choice == 'Confusion matrix':
            cm = confusion_matrix(y_res, clf.predict(X_res))
        return pd.DataFrame(cm, 
                            index=[f'Classe réelle {i}' for i in le.classes_], 
                            columns=[f'Classe prédite {i}' for i in le.classes_])
        

    # Sélection du modèle
    choix = ['Random Forest', 'XGBClassifier']  # Correction ici
    option = st.selectbox('Choix du modèle', choix)
    st.write('Le modèle choisi est :', option)

    # Entraînement et affichage des résultats
    clf = prediction(option)
    if clf is not None:  # Vérifie que le modèle a bien été créé
        display = st.radio('Que souhaitez-vous montrer ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(f"Accuracy: {scores(clf, display):.2f}")
    elif display == 'Confusion matrix':
        st.write("Matrice de confusion :")
        st.dataframe(scores(clf, display))

# Page Conclusion
def page_4():
    st.title("Conclusion")
    st.write("L’analyse des données des feux de forêt aux USA a permis de comprendre quelles sont les critical data que sont les classes de feu par état, les régions dont le climat et la température sont favorables au déclenchement des feux.")
    st.write("Suite au traitement et à l’analyse de ces données, nous avons pu mettre en place une modélisation permettant ainsi de prédire les classes de feu par état, par période via la température moyenne relevée sur chacun des états.")
    st.write("Les modèles retenus sont le random forest et le XGClassifier. Nous avons par la suite affiné ces modèles en réalisant un regroupement des données critiques, d’un échantillonnage de ces données et une mise en place d’une feature importance.")
    st.write("Ceci a permis d’obtenir des modèles performants permettant ainsi de prédire les feux.")
    st.write("Cependant ces modèles peuvent être encore améliorés en intégrant de nouveaux types de données comme l’imagerie des zones à risque par drone/satellite avec des données en 3 dimensions par exemple, de meilleurs relevés météorologiques (vents, nombre de précipitations,...).")

# Appel des pages en fonction de la navigation
if menu == "Accueil":
    page_accueil()
elif menu == "Exploration data":
    page_1()
elif menu == "Data Visualization":
    page_2()
elif menu == "Modélisation":
    page_3()
elif menu == "Conclusion":
    page_4()
  



