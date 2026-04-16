import os
import torch
import data_utils
from clap import core as clap_utils

from tqdm import tqdm
from torch.utils.data import DataLoader

PM_SUFFIX = {"max":"_max", "avg":""}


def save_backbone_audio_features(target_model, dataset, save_name, batch_size=256, device="cuda"):
    """Save backbone audio features for a manifest-backed audio dataset."""
    _make_save_dir(save_name)
    if os.path.exists(save_name):
        return

    # feature extractor is frozen for CBM training
    target_model.eval()
    for param in target_model.parameters():
        param.requires_grad = False

    all_features = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size, num_workers=2, pin_memory=True)):
            audio = batch["audio"].to(device)
            if hasattr(target_model, "expects_sample_rates") and target_model.expects_sample_rates:
                features = target_model(audio, sample_rates=batch["sr"])
            else:
                features = target_model(audio)
            if features.dim() > 2:
                features = torch.flatten(features, start_dim=1)
            all_features.append(features.detach().cpu())

    torch.save(torch.cat(all_features, dim=0), save_name)
    del all_features
    torch.cuda.empty_cache()


def save_clap_audio_features(clap_bundle, dataset, save_name, batch_size=128):
    """Save CLAP audio embeddings for each sample in a dataset."""
    _make_save_dir(save_name)
    if os.path.exists(save_name):
        return

    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(dataset, batch_size, num_workers=2, pin_memory=True)):
            embs = clap_utils.encode_audio(
                audio_or_paths=batch["audio"],
                sample_rates=batch["sr"].tolist(),
                clap_bundle=clap_bundle,
                batch_size=batch_size,
                normalize=True,
            )
            all_embeddings.append(embs)

    torch.save(torch.cat(all_embeddings, dim=0), save_name)
    del all_embeddings
    torch.cuda.empty_cache()


def save_clap_text_features(clap_bundle, concepts, save_name, batch_size=128):
    """Save CLAP text embeddings for concept strings."""
    _make_save_dir(save_name)
    if os.path.exists(save_name):
        return
    embs = clap_utils.encode_text(concepts=concepts, clap_bundle=clap_bundle, batch_size=batch_size, normalize=True)
    torch.save(embs, save_name)
    del embs
    torch.cuda.empty_cache()


def get_audio_save_names(clap_model_name, target_name, target_layer, split, concept_set, pool_mode, save_dir):
    clap_tag = clap_model_name.replace("/", "_")
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    target_save_name = "{}/{}_backbone_{}_{}{}.pt".format(
        save_dir,
        split,
        target_name.replace("/", "_"),
        target_layer,
        PM_SUFFIX[pool_mode],
    )
    clap_audio_save_name = "{}/{}_clap_audio_{}.pt".format(save_dir, split, clap_tag)
    clap_text_save_name = "{}/{}_clap_text_{}.pt".format(save_dir, concept_set_name, clap_tag)
    return target_save_name, clap_audio_save_name, clap_text_save_name


def get_audio_split_cache_key(dataset_name, split, hf_subset=None):
    if dataset_name == "audioset":
        return data_utils.get_hf_audioset_cache_key(split=split, subset=hf_subset)
    return split


def save_audio_activations(clap_model_name, target_name, target_layers, dataset_name, split,
                           concept_set, batch_size, device, pool_mode, save_dir,
                           hf_streaming=False, hf_cache_dir=None, max_items=None, hf_subset=None):
    """Save backbone audio features and CLAP audio/text embeddings for a split."""
    split_cache_key = get_audio_split_cache_key(dataset_name, split, hf_subset=hf_subset)
    target_save_name, clap_audio_save_name, clap_text_save_name = get_audio_save_names(
        clap_model_name,
        target_name,
        target_layers[0],
        split_cache_key,
        concept_set,
        pool_mode,
        save_dir,
    )
    save_names = {
        "backbone": target_save_name,
        "clap_audio": clap_audio_save_name,
        "clap_text": clap_text_save_name,
    }
    if _all_saved(save_names):
        return

    clap_bundle = clap_utils.load_clap_model(model_name=clap_model_name, device=device)
    audio_dataset = data_utils.get_audio_dataset(
        dataset_name=dataset_name,
        split=split,
        hf_streaming=hf_streaming if dataset_name == "audioset" else False,
        hf_cache_dir=hf_cache_dir,
        max_items=max_items,
        hf_subset=hf_subset,
    )

    with open(concept_set, "r", encoding="utf-8") as f:
        concepts = [line.strip() for line in f.readlines() if line.strip()]

    save_clap_text_features(clap_bundle, concepts, clap_text_save_name, batch_size=batch_size)
    save_clap_audio_features(clap_bundle, audio_dataset, clap_audio_save_name, batch_size=batch_size)

    if target_name == "clap_audio":
        # Keep backbone/concept targets logically separate while allowing a CLAP-only fallback.
        if not os.path.exists(target_save_name):
            torch.save(torch.load(clap_audio_save_name, map_location="cpu"), target_save_name)
    else:
        target_model, _ = data_utils.get_target_model(target_name, device)
        save_backbone_audio_features(target_model, audio_dataset, target_save_name, batch_size=batch_size, device=device)


def compute_concept_matrix_from_activations(clap_audio_save_name, clap_text_save_name):
    clap_audio_embs = torch.load(clap_audio_save_name, map_location="cpu").float()
    clap_text_embs = torch.load(clap_text_save_name, map_location="cpu").float()
    concept_matrix = clap_utils.compute_audio_text_similarity(clap_audio_embs, clap_text_embs, normalize=True)
    if concept_matrix.dim() != 2:
        raise ValueError("Expected concept matrix to be 2D, got {}".format(tuple(concept_matrix.shape)))
    if concept_matrix.shape[0] != clap_audio_embs.shape[0] or concept_matrix.shape[1] != clap_text_embs.shape[0]:
        raise ValueError(
            "Concept matrix shape mismatch: P={}, audio_embs={}, text_embs={}".format(
                tuple(concept_matrix.shape), tuple(clap_audio_embs.shape), tuple(clap_text_embs.shape)
            )
        )
    return concept_matrix

    
def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

def get_accuracy_cbm(model, dataset, device, batch_size=250, num_workers=2):
    correct = 0
    total = 0
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                           pin_memory=True)):
        with torch.no_grad():
            #outs = target_model(images.to(device))
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu()==labels)
            total += len(labels)
    return correct/total

def get_preds_cbm(model, dataset, device, batch_size=250, num_workers=2):
    preds = []
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                           pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    return preds

def get_concept_act_by_pred(model, dataset, device):
    preds = []
    concept_acts = []
    for images, labels in tqdm(DataLoader(dataset, 500, num_workers=8, pin_memory=True)):
        with torch.no_grad():
            outs, concept_act = model(images.to(device))
            concept_acts.append(concept_act.cpu())
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    concept_acts = torch.cat(concept_acts, dim=0)
    concept_acts_by_pred=[]
    for i in range(torch.max(pred)+1):
        concept_acts_by_pred.append(torch.mean(concept_acts[preds==i], dim=0))
    concept_acts_by_pred = torch.stack(concept_acts_by_pred, dim=0)
    return concept_acts_by_pred
