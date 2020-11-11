import numpy as np 
import sys
import pandas as pd
import argparse
import numpy as np

import sklearn.metrics as metrics
from sklearn.metrics.pairwise import paired_distances
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

def read_xv(filename):
    """ Lit les xv d'un fichier et les retourne sous forme de dico """
    xv = {}
    with open(filename) as fp:
        for line in fp:
            _ = line.strip().split(' ', 1)
            xv[_[0]] = _[1].split(' ')
    return xv

def read_trial(filename): 
    """ lit un fichier de trial et le retourne sous forme de dataframe """
    def tar2int(s):
        """ Transform 'target' into 1 and 'nontarget' into 0 """
        assert s in ["target", "nontarget"], f"Unkown target type {s}"
        r = 1 if s == "target" else 0
        return r

    df = pd.read_csv(filename, delimiter=" ", header=None, names=["utt1", "utt2", "target"])
    df['target'] = df['target'].apply(tar2int)

    return df

def eer_from_ers(fpr, tpr):
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer

if __name__=="__main__":        
    parser = argparse.ArgumentParser(description='Score a trial file given enroll and test xvectors')
    parser.add_argument("--trial", type=str, required=True, help="The trial file")
    parser.add_argument("--enroll", type=str, required=True, help="The enrollment file")
    parser.add_argument("--test", type=str, required=True, help="The test file")
    args = parser.parse_args()

    # lecutre du fichier de trials
    trials = read_trial(args.trial)
    veri_utt1 = trials['utt1']
    veri_utt2 = trials['utt2']
    veri_labs = np.asarray(trials['target'], dtype=int)
    
    # lecture et mise en forme des xv
    all_xv = read_xv(args.test)
    all_xv.update(read_xv(args.enroll))

    all_embeds = np.array(list(all_xv.values()))
    all_embeds = np.vstack(all_embeds)
    all_embeds = normalize(all_embeds, axis=1)

    all_utts = np.array(list(all_xv.keys()))

    utt_embed = {k:v for k, v in zip(all_utts, all_embeds)}

    emb0 = np.array([utt_embed[k] for k in veri_utt1])
    emb1 = np.array([utt_embed[k] for k in veri_utt2])

    # calcul des scores et taux d'erreur
    scores = paired_distances(emb0, emb1, metric='cosine')
    fpr, tpr, thresholds = metrics.roc_curve(1 - veri_labs, scores, pos_label=1, drop_intermediate=False)
    
    # plot des taux d'erreurs
    plt.title("Taux d'erreur en fonction du seuil")
    plt.plot(thresholds, 1 - tpr, 'b', label="False negative")
    plt.plot(thresholds, fpr, 'r', label="False positive")
    plt.plot(thresholds, tpr - fpr, 'g', label="tpr - fpr")
    plt.legend(loc = 'lower right')
    plt.ylabel('%')
    plt.xlabel('Threshold')
    plt.show()
    
    # print de l'EER et du threshold
    seuil = thresholds[np.argmax(tpr-fpr)]
    print(f"Threshold = {seuil:.4f}")

    eer = eer_from_ers(fpr, tpr)*100
    print(f'eer : {eer}')

    # Accuracy
    acc = ((scores < seuil) == veri_labs).sum() / len(veri_labs)
    acc *= 100
    print(f"Accuracy : {acc:.2f}%")
