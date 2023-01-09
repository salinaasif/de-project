# Sentiment analysis
## Objectif du projet

L’idée générale de ce projet est de proposer un modèle de Machine Learning permettant d'analyser le sentiment d'une phrase associée au domaine de la finance. Pour cela, nous avons à notre disposition un jeu de données d'entraînement composé de 4840 phrases dans lequel chacune des ces dernières est associée à un sentiment (1 : négatif ; 2 : positif). Par conséquent, le but de notre travail est de permettre à l’utilisateur de saisir une phrase à la suite de laquelle une sortie sera envoyée avec le sentiment associé. 

## Fichier mis à disposition
Au sein de ce repository git, vous trouverez :

- la base de données utilisée pour entrainer notre modèle ;
- le code source dans lequel la phase de pre-process et la mise en place de notre modèle (Réseau de neurone) sont réalisées ;
- le dockerfile ;
- le fichier requirements.txt.

## Lancement du conteneur
Afin de pouvoir saisir une phrase et obtenir le sentiment associé, plusieurs commandes sont à exécuter dans le terminal :

1) docker pull jlydie/de-project:first	-> récupération de l’image depuis Docker Hub
2) docker run -ti jlydie/de-project:first -> run le conteneur afin de saisir une phrase en input.
