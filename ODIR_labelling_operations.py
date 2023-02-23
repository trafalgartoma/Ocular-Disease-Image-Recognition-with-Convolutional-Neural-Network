import csv
import os
import pandas as pd

# path dei file csv utilizzati
path_csv = "/Users/giuse/Desktop/ML_in_Health_Applications/base_data/dataset_base.csv"
path_csv_finale = "/Users/giuse/Desktop/file_labels_finale.csv"
conta_img = 0
'''
# blacklist delle immagini da scartare
def is_blacklisted_image(image_name):
    # creo una blacklist delle immagini che hanno difetti nella foto
    blacklist = {'2174_right.jpg', '2175_left.jpg', '2176_left.jpg', '2177_left.jpg', '2177_right.jpg',
                 '2178_right.jpg', '2179_left.jpg', '2179_right.jpg', '2180_left.jpg', '2180_right.jpg',
                 '2181_left.jpg', '2181_right.jpg', '2182_left.jpg', '2182_right.jpg', '2957_left.jpg',
                 '2957_right.jpg'}

    return image_name in blacklist
'''

# blacklist delle keywords sa non processare
def is_blacklisted_keyword(keyword):
    blacklist = {"anterior segment image", "no fundus image", "lens dust", "optic disk photographically invisible",
                 "low image quality", "image offset", "wrong background"}
    return keyword in blacklist
 # termini che verranno labellizzati con O

#Keywords per other
def other(keyword):
    O_keywords = {
    'macular epiretinal membrane',
    'epiretinal membrane',
    'drusen',
    'myelinated nerve fibers',
    'laser spot',
    'vitreous degeneration',
    'refractive media opacity',
    'spotted membranous change',
    'tessellated fundus',
    'maculopathy',
    'chorioretinal atrophy',
    'branch retinal vein occlusion',
    'retinal pigmentation',
    'white vessel',
    'post retinal laser surgery',
    'epiretinal membrane over the macula',
    'retinitis pigmentosa',
    'central retinal vein occlusion',
    'optic disc edema',
    'post laser photocoagulation',
    'retinochoroidal coloboma',
    'atrophic change',
    'optic nerve atrophy',
    'old branch retinal vein occlusion',
    'depigmentation of the retinal pigment epithelium',
    'chorioretinal atrophy with pigmentation proliferation',
    'central retinal artery occlusion',
    'old chorioretinopathy',
    'pigment epithelium proliferation',
    'retina fold',
    'abnormal pigment ',
    'idiopathic choroidal neovascularization',
    'branch retinal artery occlusion',
    'vessel tortuosity',
    'pigmentation disorder',
    'rhegmatogenous retinal detachment',
    'macular hole',
    'morning glory syndrome',
    'atrophy',
    'arteriosclerosis',
    'asteroid hyalosis',
    'congenital choroidal coloboma',
    'macular coloboma',
    'optic discitis',
    'oval yellow-white atrophy',
    'wedge-shaped change',
    'wedge white line change',
    'retinal artery macroaneurysm',
    'retinal vascular sheathing',
    'suspected abnormal color of  optic disc',
    'suspected retinal vascular sheathing',
    'suspected retinitis pigmentosa',
    'silicone oil eye',
    'fundus laser photocoagulation spots',
    'glial remnants anterior to the optic disc',
    'intraretinal microvascular abnormality'
    }
    return keyword in O_keywords


# funzione che serve a identificare le keywords e le inserisce in una lista
def process_keywords(img_name, keywords):
    listkeywords = [x.strip() for x in keywords.split('，')]
    normal = 0
    diabetes = 0
    glaucoma = 0
    cataract = 0
    amd = 0
    hypertension = 0
    myopia = 0
    others = 0
    not_decisive = 0

    # processiamo le keyword
    for keyword in listkeywords:
        if "normal fundus" in keyword:
            normal = 1
        elif "diabetic retinopathy" in keyword or "proliferative retinopathy" in keyword:
            diabetes = 1
        elif "glaucoma" in keyword:
            glaucoma = 1
        elif "cataract" in keyword:
            cataract = 1
        elif "macular degeneration" in keyword:
            amd = 1
        elif "hypertensive retinopathy" in keyword:
            hypertension = 1
        elif "myopi" in keyword:
            myopia = 1
        elif other(keyword):
            others = 1
        else:
            # se la keyword appartiene alla blacklist creata per la label other
            if is_blacklisted_keyword(keyword):
                not_decisive = 1
            # azzeriamo i valori in return per poi andare a eliminare la riga interessata
    if not_decisive == 1:
        normal = 0
        diabetes = 0
        glaucoma = 0
        cataract = 0
        amd = 0
        hypertension = 0
        myopia = 0
        others = 0

    return [img_name, normal, diabetes, glaucoma, cataract, amd, hypertension, myopia, others]


def all_same(items):
    return all(x == items[1] for x in items[1:])


# engine = RuleEngine()

# apriamo il file csv e scriviamo la prima riga
with open(path_csv_finale, 'w', newline='') as csv_file_fin:
    file_writer = csv.writer(csv_file_fin, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    file_writer.writerow(['ID', 'Normal', 'Diabetes', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension',
                          'Myopia', 'Others'])

    # creiamo un dataframe per contenere momentaneamente i dati del xlsx

    # leggiamo il csv
    df = pd.read_csv(path_csv, sep=';')

    # impostiamo delle variabili che andremo ad utilizzare
    decisive = 0
    totale_img_skip = 0;

    # per ogni riga del dataframe si va ad analizzare ogni info inerente ad ogni occhio
    for row in df.itertuples():
        # creiamo una variabile per contenere le informazioni di una riga
        tuple = row;
        # operazioni sui left fundus
        if tuple[4].find("Left-Fundus"):
            temp1 = process_keywords(tuple[4], tuple[6])
            if all_same(temp1):
                print("immagine skippata", temp1)
                totale_img_skip =  totale_img_skip + 1
            else:
                print(temp1)
                conta_img = conta_img + 1
                file_writer.writerow(temp1)
        # operazioni sui right fundus
        if tuple[5].find("Right-Fundus"):
            temp2 = process_keywords(tuple[5], tuple[7])
            if all_same(temp2):
                print("immagine skippata", temp2)
                totale_img_skip = totale_img_skip + 1
            else:
                conta_img = conta_img + 1
                file_writer.writerow(temp2)


print("immagini totali",conta_img)
print("immagini skippate",totale_img_skip)
