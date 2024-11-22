"""Constants for `process_camme.py`"""

# * Survey waves to ignore
IGNORE_SUPPLEMENTS = ["be", "cnle", "cov", "pf"]
IGNORE_HOUSING = "log"
# Only these years had separate housing surveys
IGNORE_HOUSING_YEARS = ["2016", "2017"]

# * Variables and corresponding column names (change over course of series)
# * Variables used in Andrade et al. (2023) and others of interest
VARS_DICT = {
    ## Date
    "month": {
        "1989": "VALVAG",  # ! mois de réalisation de l'enquête
        "1991": "valvag",
        "2004": "MOISENQ",
    },
    ## From Andrade et al. (2023)
    "inf_exp_qual": {
        "1989": "QUEST_GEN_5",  # ! Evolution des prix dans les prochains mois
        "1991": "q5",
        "2004": "EVOLPRIX",  # Par rapport aux douze derniers mois, quelle sera  à votre avis l’évolution des prix au cours  des douze prochains mois ?
    },
    "inf_exp_val_inc": {
        "1989": "",
        "1991": "",
        "2004": "EVPRIPLU",
    },
    "inf_exp_val_dec": {
        "1989": "",
        "1991": "",
        "2004": "EVPRIBAI",
    },
    "consump_past": {
        "1989": "QUEST_GEN_12",  # ! Réalisation d'achats de biens d'équipement du ménage depuis un an
        "1991": "q12",
        "2004": "EQUIPPAS",  # Avez-vous fait des achats importants au cours des douze derniers mois ? (machine à laver, réfrigérateur, meubles, lave-vaisselle, ...)
    },
    "consump_general": {
        "1989": "QUEST_GEN_6",  # ! Opportunité de faire des achats importants actuellement (meubles, machines à laver, télévision, ...)
        "1991": "q6",
        "2004": "ACHATS",  # Dans la situation économique actuelle, pensez-vous que les gens aient intérêt à faire des achats importants ? (meubles, machines à laver, matériels électroniques ou informatiques...)
    },
    ## Others
    "spend_change": {
        "1989": "QUEST_GEN_14",  # ! Intention de dépenses en biens d'équipement l'année à venir par rapport à l'année passée
        "1991": "q14",
        "2004": "DEPENSES",  # Au cours des douze prochains mois, par rapport aux douze mois passés, avez-vous l'intention de dépenser, pour effectuer des achats importants…
    },
    "econ_exp": {
        "1989": "",
        "1991": "",
        "2004": "ECOFUT",  # A votre avis, au cours des douze prochains mois, la situation économique générale de la France va …
    },
    "personal_save_fut": {
        "1989": "QUEST_GEN_11",  # ! Capacité du ménage à épargner dans les prochains mois
        "1991": "q11",
        "2004": "ECONOMIS",  # Pensez-vous réussir à mettre de l'argent de côté au cours des douze prochains mois ?
    },
    "general_save": {
        "1989": "QUEST_GEN_7",  # ! Opportunité d'épargner
        "1991": "q7",
        "2004": "EPARGNER",  # Dans la situation économique actuelle, pensez-vous que ce soit le bon moment pour épargner ?
    },
    "personal_spend_exp": {
        "1989": "QUEST_GEN_13",  # ! Intentions d'achats de biens d'équipement du ménage d'ici un an
        "1991": "q13",
        "2004": "EQUIPFUT",  # Avez-vous l'intention d'effectuer des achats importants  au cours des douze prochains mois ?
    },
    "inf_per_qual": {
        "1989": "QUEST_GEN_4",  # ! Evolution des prix depuis six mois
        "1991": "q4",
        "2004": "PRIX",  # Trouvez-vous que, au cours des douze derniers mois, les prix ont …
    },
    "inf_per_val_inc": {
        "1989": "",
        "1991": "",
        "2004": "PRIXPLUS",
    },
    "inf_per_val_dec": {
        "1989": "",
        "1991": "",
        "2004": "PRIXBAIS",
    },
}
