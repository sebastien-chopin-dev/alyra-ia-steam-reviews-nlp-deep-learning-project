# Stratégies Deep learning reviews Steam Jeux vidéo

## Les modèles
- BERT base (anglais)
- BERT tweet (anglais)

- CamemBERT (français)
- XLM-RoBERTa (anglais et français)

## Dataset à process à partir du fichier des 100 millions multi-langue
- Anglais avec weighted_score > 0.8 + filtre légitime (longueur entre 5 et 500 mots, filtre ascii art) -> 25 000 (équilibré)
- Anglais et français weighted_score > 0.6 + filtre légitime (longueur entre 5 et 500 mots, filtre ascii art)

## Fine tuning
- Fine tuning - exit layer uniquement
- Fine tuning - all layer

## Distillation sur le meilleur anglais

## Structure du projet

### Notebooks

- exploration : analyse fichier weighted_score > 0.8 et élargir weighted_score > 0.5, filtrer pour garder reviews légitimes

- fine tuning dataset anglais modèle base et tweet, comparaison
- fine tuning dataset multilangue modèle XLM-RoBERTa et CamemBERT, comparaison

- distillation modèle anglais

### Scripts

- data_preparation.py : préparation du dataset pour fine tuning.  Paramétrable (longueur des reviews, feature de contexte à prendre en compte)