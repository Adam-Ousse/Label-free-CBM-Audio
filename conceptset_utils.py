import random
import numpy as np
import torch

import clap_utils
from sentence_transformers import SentenceTransformer


_MPNET_MODEL = None
_CLAP_BUNDLES = {}


def _get_mpnet_model():
    global _MPNET_MODEL
    if _MPNET_MODEL is None:
        _MPNET_MODEL = SentenceTransformer("all-mpnet-base-v2")
    return _MPNET_MODEL


def _get_clap_bundle(clap_model, device):
    key = (clap_model, device)
    if key not in _CLAP_BUNDLES:
        _CLAP_BUNDLES[key] = clap_utils.load_clap_model(model_name=clap_model, device=device)
    return _CLAP_BUNDLES[key]


def remove_too_long(concepts, max_len, print_prob=0):
    """
    deletes all concepts longer than max_len
    """
    new_concepts = []
    for concept in concepts:
        if len(concept) <= max_len:
            new_concepts.append(concept)
        else:
            if random.random()<print_prob:
                print(len(concept), concept)
    print(len(concepts), len(new_concepts))
    return new_concepts


def filter_too_similar_to_cls(concepts, classes, sim_cutoff, device="cuda", print_prob=0):
    #first check simple text matches
    print(len(concepts))
    concepts = list(concepts)
    concepts = sorted(concepts)
    
    for cls in classes:
        for prefix in ["", "a ", "A ", "an ", "An ", "the ", "The "]:
            try:
                concepts.remove(prefix+cls)
                if random.random()<print_prob:
                    print("Class:{} - Deleting {}".format(cls, prefix+cls))
            except(ValueError):
                pass
        try:
            concepts.remove(cls.upper())
        except(ValueError):
            pass
        try:
            concepts.remove(cls[0].upper()+cls[1:])
        except(ValueError):
            pass
    print(len(concepts))
        
    mpnet_model = _get_mpnet_model()
    class_features_m = mpnet_model.encode(classes)
    concept_features_m = mpnet_model.encode(concepts)
    dot_prods_m = class_features_m @ concept_features_m.T
    dot_prods_c = _clap_text_dot_prods(classes, concepts, device=device)
    #weighted since mpnet has highger variance
    dot_prods = (dot_prods_m + 3*dot_prods_c)/4
    
    to_delete = []
    for i in range(len(classes)):
        for j in range(len(concepts)):
            prod = dot_prods[i,j]
            if prod >= sim_cutoff and i!=j:
                if j not in to_delete:
                    to_delete.append(j)
                    if random.random()<print_prob:
                        print("Class:{} - Concept:{}, sim:{:.3f} - Deleting {}".format(classes[i], concepts[j], dot_prods[i,j], concepts[j]))
                        
    to_delete = sorted(to_delete)[::-1]

    for item in to_delete:
        concepts.pop(item)
    print(len(concepts))
    return concepts

def filter_too_similar(concepts, sim_cutoff, device="cuda", print_prob=0):
    
    mpnet_model = _get_mpnet_model()
    concept_features = mpnet_model.encode(concepts)
        
    dot_prods_m = concept_features @ concept_features.T
    dot_prods_c = _clap_text_dot_prods(concepts, concepts, device=device)
    
    dot_prods = (dot_prods_m + 3*dot_prods_c)/4
    
    to_delete = []
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            prod = dot_prods[i,j]
            if prod >= sim_cutoff and i!=j:
                if i not in to_delete and j not in to_delete:
                    to_print = random.random() < print_prob
                    #Deletes the concept with lower average similarity to other concepts - idea is to keep more general concepts
                    if np.sum(dot_prods[i]) < np.sum(dot_prods[j]):
                        to_delete.append(i)
                        if to_print:
                            print("{} - {} , sim:{:.4f} - Deleting {}".format(concepts[i], concepts[j], dot_prods[i,j], concepts[i]))
                    else:
                        to_delete.append(j)
                        if to_print:
                            print("{} - {} , sim:{:.4f} - Deleting {}".format(concepts[i], concepts[j], dot_prods[i,j], concepts[j]))
                            
    to_delete = sorted(to_delete)[::-1]
    for item in to_delete:
        concepts.pop(item)
    print(len(concepts))
    return concepts


def _clap_text_dot_prods(list1, list2, device="cuda", clap_model="laion/clap-htsat-unfused", batch_size=256):
    "Returns: numpy array with CLAP text-space dot products"
    clap_bundle = _get_clap_bundle(clap_model=clap_model, device=device)
    features1 = clap_utils.encode_text(list1, clap_bundle=clap_bundle, batch_size=batch_size, normalize=True)
    features2 = clap_utils.encode_text(list2, clap_bundle=clap_bundle, batch_size=batch_size, normalize=True)
    dot_prods = features1 @ features2.T
    return dot_prods.cpu().numpy()

def most_similar_concepts(word, concepts, device="cuda"):
    """
    returns most similar words to a given concepts
    """
    mpnet_model = _get_mpnet_model()
    word_features = mpnet_model.encode([word])
    concept_features = mpnet_model.encode(concepts)
        
    dot_prods_m = word_features @ concept_features.T
    dot_prods_c = _clap_text_dot_prods([word], concepts, device=device)
    
    dot_prods = (dot_prods_m + 3*dot_prods_c)/4
    min_distance, indices = torch.topk(torch.FloatTensor(dot_prods[0]), k=5)
    return [(concepts[indices[i]], min_distance[i]) for i in range(len(min_distance))]