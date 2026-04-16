import torch
import os
import random
import utils
import data_utils
import similarity
import argparse
import datetime
import json
import torch.nn.functional as F

from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from torch.utils.data import DataLoader, TensorDataset

parser = argparse.ArgumentParser(description='Settings for creating CBM')


parser.add_argument("--dataset", type=str, default="esc50")
parser.add_argument("--concept_set", type=str, default=None, 
                    help="path to concept set name")
parser.add_argument("--backbone", type=str, default="clap_audio", help="Which pretrained model to use as backbone")
parser.add_argument("--clap_model", type=str, default="laion/clap-htsat-unfused", help="Which CLAP model to use for concept scoring")

parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")
parser.add_argument("--saga_batch_size", type=int, default=256, help="Batch size used when fitting final layer")
parser.add_argument("--proj_batch_size", type=int, default=50000, help="Batch size to use when learning projection layer")

parser.add_argument("--feature_layer", type=str, default='layer4', 
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLAP activations")
parser.add_argument("--save_dir", type=str, default='saved_models', help="where to save trained models")
parser.add_argument("--clip_cutoff", type=float, default=0.25, help="Deprecated alias: use --concept_activation_cutoff")
parser.add_argument("--concept_activation_cutoff", type=float, default=None, help="concepts with smaller top5 CLAP activation will be deleted")
parser.add_argument("--proj_steps", type=int, default=1000, help="how many steps to train the projection layer for")
parser.add_argument("--interpretability_cutoff", type=float, default=0.45, help="concepts with smaller similarity to target concept will be deleted")
parser.add_argument("--lam", type=float, default=0.0007, help="Sparsity regularization parameter, higher->more sparse")
parser.add_argument("--n_iters", type=int, default=1000, help="How many iterations to run the final layer solver for")
parser.add_argument("--print", action='store_true', help="Print all concepts being deleted in this stage")
parser.add_argument("--train_split", type=str, default=None, help="Train split for audio datasets")
parser.add_argument("--val_split", type=str, default=None, help="Validation split for audio datasets")
parser.add_argument("--test_split", type=str, default=None, help="Optional test split for post-training evaluation")
parser.add_argument("--enforce_esc50_fold1_protocol", action='store_true', help="Use fold1_test/fold1_val/fold1_train protocol for ESC-50")
parser.add_argument("--audioset_streaming", action='store_true', help="Use Hugging Face streaming for AudioSet")
parser.add_argument("--audioset_cache_dir", type=str, default=None, help="Optional Hugging Face cache directory")
parser.add_argument("--audioset_max_items", type=int, default=None, help="Optional cap on loaded AudioSet samples")
parser.add_argument("--audioset_subset", type=str, default=None, help="Default AudioSet subset (balanced, unbalanced, full)")
parser.add_argument("--audioset_train_subset", type=str, default=None, help="AudioSet subset for train split")
parser.add_argument("--audioset_val_subset", type=str, default=None, help="AudioSet subset for val split")
parser.add_argument("--audioset_test_subset", type=str, default=None, help="AudioSet subset for test split")


def _labels_to_multihot_tensor(labels, num_classes):
    target = torch.zeros(num_classes, dtype=torch.float32)
    for label in labels or []:
        try:
            idx = int(label)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < num_classes:
            target[idx] = 1.0
    return target


def _compute_multilabel_metrics(logits, targets, threshold=0.5):
    metrics = {}

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()
    y_pred = (probs >= float(threshold)).astype("int32")

    try:
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    except Exception:
        metrics["warning"] = "scikit-learn unavailable; only loss was computed"
        return metrics

    try:
        metrics["mAP_macro"] = float(average_precision_score(y_true, probs, average="macro"))
    except ValueError:
        metrics["mAP_macro"] = None

    try:
        metrics["auc_roc_macro"] = float(roc_auc_score(y_true, probs, average="macro"))
    except ValueError:
        metrics["auc_roc_macro"] = None

    try:
        metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    except ValueError:
        metrics["f1_micro"] = None
        metrics["f1_macro"] = None

    metrics["threshold"] = float(threshold)
    return metrics

def train_cbm_and_save(args):
    
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.concept_set==None:
        args.concept_set = "data/concept_sets/{}_filtered.txt".format(args.dataset)

    if args.concept_activation_cutoff is None:
        args.concept_activation_cutoff = args.clip_cutoff
        
    similarity_fn = similarity.cos_similarity_cubed_single
    
    if args.dataset == "audioset":
        d_train = args.train_split or "train"
        d_val = args.val_split or "eval"
        d_test = args.test_split

        hf_subset_default = getattr(args, "audioset_subset", None)
        hf_train_subset = getattr(args, "audioset_train_subset", None) or hf_subset_default
        hf_val_subset = getattr(args, "audioset_val_subset", None) or hf_subset_default
        hf_test_subset = getattr(args, "audioset_test_subset", None) or hf_subset_default

        audio_probe_specs = [
            ("train", d_train, hf_train_subset),
            ("val", d_val, hf_val_subset),
        ]
        if d_test is not None:
            audio_probe_specs.append(("test", d_test, hf_test_subset))
    elif args.dataset in {"esc50", "urbansound8k"}:
        if args.dataset == "esc50" and args.enforce_esc50_fold1_protocol:
            fixed_train = "fold1_train"
            fixed_val = "fold1_val"
            fixed_test = "fold1_test"
            if args.train_split is not None and args.train_split != fixed_train:
                raise ValueError("ESC-50 fold protocol requires train_split='{}'".format(fixed_train))
            if args.val_split is not None and args.val_split != fixed_val:
                raise ValueError("ESC-50 fold protocol requires val_split='{}'".format(fixed_val))
            if args.test_split is not None and args.test_split != fixed_test:
                raise ValueError("ESC-50 fold protocol requires test_split='{}'".format(fixed_test))
            d_train = fixed_train
            d_val = fixed_val
            d_test = fixed_test
        else:
            if args.dataset == "urbansound8k":
                d_train = args.train_split or "fold10_train"
                d_val = args.val_split or "fold10_val"
                d_test = args.test_split or "fold10_test"
            else:
                d_train = args.train_split or "train"
                d_val = args.val_split or "val"
                d_test = args.test_split
        audio_probe_specs = [
            ("train", d_train, None),
            ("val", d_val, None),
        ]
        if d_test is not None:
            audio_probe_specs.append(("test", d_test, None))
    else:
        raise ValueError(
            "Unsupported dataset '{}' for audio-only runtime. Use esc50, urbansound8k, or audioset.".format(
                args.dataset
            )
        )

    print("Using splits -> train: {}, val: {}, test: {}".format(d_train, d_val, d_test))
    
    # classes for audio datasets
    classes = data_utils.get_dataset_classes(args.dataset)
    
    with open(args.concept_set, "r", encoding="utf-8") as f:
        concepts = [line.strip() for line in f.readlines() if line.strip()]
    
    #save activations and get save_paths
    hf_streaming = bool(getattr(args, "audioset_streaming", False))
    hf_cache_dir = getattr(args, "audioset_cache_dir", None)
    hf_max_items = getattr(args, "audioset_max_items", None)

    for _, d_probe, d_probe_subset in audio_probe_specs:
        utils.save_audio_activations(
            clap_model_name=args.clap_model,
            target_name=args.backbone,
            target_layers=[args.feature_layer],
            dataset_name=args.dataset,
            split=d_probe,
            concept_set=args.concept_set,
            batch_size=args.batch_size,
            device=args.device,
            pool_mode="avg",
            save_dir=args.activation_dir,
            hf_streaming=hf_streaming,
            hf_cache_dir=hf_cache_dir,
            max_items=hf_max_items,
            hf_subset=d_probe_subset,
        )

    train_cache_key = utils.get_audio_split_cache_key(args.dataset, d_train, hf_subset=audio_probe_specs[0][2])
    val_cache_key = utils.get_audio_split_cache_key(args.dataset, d_val, hf_subset=audio_probe_specs[1][2])
    test_cache_key = None
    if d_test is not None:
        test_cache_key = utils.get_audio_split_cache_key(args.dataset, d_test, hf_subset=audio_probe_specs[2][2])

    target_save_name, clap_audio_save_name, clap_text_save_name = utils.get_audio_save_names(
        args.clap_model,
        args.backbone,
        args.feature_layer,
        train_cache_key,
        args.concept_set,
        "avg",
        args.activation_dir,
    )
    val_target_save_name, val_clap_audio_save_name, clap_text_save_name = utils.get_audio_save_names(
        args.clap_model,
        args.backbone,
        args.feature_layer,
        val_cache_key,
        args.concept_set,
        "avg",
        args.activation_dir,
    )
    if d_test is not None:
        test_target_save_name, test_clap_audio_save_name, clap_text_save_name = utils.get_audio_save_names(
            args.clap_model,
            args.backbone,
            args.feature_layer,
            test_cache_key,
            args.concept_set,
            "avg",
            args.activation_dir,
        )

    if os.path.exists(clap_text_save_name):
        cached_text_embs = torch.load(clap_text_save_name, map_location="cpu")
        cached_concept_count = int(cached_text_embs.shape[0])
        if cached_concept_count != len(concepts):
            print(
                "Detected stale CLAP text cache (cached={}, expected={}). Rebuilding text embeddings.".format(
                    cached_concept_count,
                    len(concepts),
                )
            )
            os.remove(clap_text_save_name)
            for _, d_probe, d_probe_subset in audio_probe_specs:
                utils.save_audio_activations(
                    clap_model_name=args.clap_model,
                    target_name=args.backbone,
                    target_layers=[args.feature_layer],
                    dataset_name=args.dataset,
                    split=d_probe,
                    concept_set=args.concept_set,
                    batch_size=args.batch_size,
                    device=args.device,
                    pool_mode="avg",
                    save_dir=args.activation_dir,
                    hf_streaming=hf_streaming,
                    hf_cache_dir=hf_cache_dir,
                    max_items=hf_max_items,
                    hf_subset=d_probe_subset,
                )
    
    #load features
    with torch.no_grad():
        target_features = torch.load(target_save_name, map_location="cpu").float()
        
        val_target_features = torch.load(val_target_save_name, map_location="cpu").float()
        if d_test is not None:
            test_target_features = torch.load(test_target_save_name, map_location="cpu").float()
    
        concept_matrix = utils.compute_concept_matrix_from_activations(clap_audio_save_name, clap_text_save_name)
        val_concept_matrix = utils.compute_concept_matrix_from_activations(val_clap_audio_save_name, clap_text_save_name)
        if d_test is not None:
            test_concept_matrix = utils.compute_concept_matrix_from_activations(test_clap_audio_save_name, clap_text_save_name)

        if concept_matrix.shape[1] != len(concepts):
            raise ValueError(
                "Concept count mismatch: concept_set has {} concepts but CLAP concept matrix has {} columns. "
                "Clear stale activation cache in '{}' and rerun.".format(
                    len(concepts), concept_matrix.shape[1], args.activation_dir
                )
            )

        if concept_matrix.shape[0] != target_features.shape[0]:
            raise ValueError(
                "Train shape mismatch: concept_matrix N={} vs target_features N={}".format(
                    concept_matrix.shape[0], target_features.shape[0]
                )
            )
        if val_concept_matrix.shape[0] != val_target_features.shape[0]:
            raise ValueError(
                "Val shape mismatch: concept_matrix N={} vs target_features N={}".format(
                    val_concept_matrix.shape[0], val_target_features.shape[0]
                )
            )
        if d_test is not None and test_concept_matrix.shape[0] != test_target_features.shape[0]:
            raise ValueError(
                "Test shape mismatch: concept_matrix N={} vs target_features N={}".format(
                    test_concept_matrix.shape[0], test_target_features.shape[0]
                )
            )
    
    #filter concepts not activating highly
    highest = torch.mean(torch.topk(concept_matrix, dim=0, k=5)[0], dim=0)
    
    if args.print:
        for i, concept in enumerate(concepts):
            if highest[i] <= args.concept_activation_cutoff:
                print("Deleting {}, CLAP top5:{:.3f}".format(concept, highest[i]))
    concepts = [concepts[i] for i in range(len(concepts)) if highest[i]>args.concept_activation_cutoff]
    if len(concepts) == 0:
        raise ValueError("No concepts survived concept activation cutoff: {}".format(args.concept_activation_cutoff))
    
    #save memory by recalculating
    del concept_matrix
    with torch.no_grad():
        concept_matrix = utils.compute_concept_matrix_from_activations(clap_audio_save_name, clap_text_save_name)
        concept_matrix = concept_matrix[:, highest>args.concept_activation_cutoff]

        if concept_matrix.shape[1] != len(concepts):
            raise ValueError(
                "Post-filter mismatch: filtered concept list has {} entries but concept matrix has {} columns.".format(
                    len(concepts), concept_matrix.shape[1]
                )
            )

    val_concept_matrix = val_concept_matrix[:, highest>args.concept_activation_cutoff]
    if d_test is not None:
        test_concept_matrix = test_concept_matrix[:, highest>args.concept_activation_cutoff]
    
    #learn projection layer
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts),
                                 bias=False).to(args.device)
    opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3)
    
    indices = [ind for ind in range(len(target_features))]
    
    best_val_loss = float("inf")
    best_step = 0
    best_weights = None
    proj_batch_size = min(args.proj_batch_size, len(target_features))
    for i in range(args.proj_steps):
        batch = torch.LongTensor(random.sample(indices, k=proj_batch_size))
        outs = proj_layer(target_features[batch].to(args.device).detach())
        loss = -similarity_fn(concept_matrix[batch].to(args.device).detach(), outs)
        
        loss = torch.mean(loss)
        loss.backward()
        opt.step()
        if i%50==0 or i==args.proj_steps-1:
            with torch.no_grad():
                val_output = proj_layer(val_target_features.to(args.device).detach())
                val_loss = -similarity_fn(val_concept_matrix.to(args.device).detach(), val_output)
                val_loss = torch.mean(val_loss)
            if i==0:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
                print("Step:{}, Avg train similarity:{:.4f}, Avg val similarity:{:.4f}".format(best_step, -loss.cpu(),
                                                                                               -best_val_loss.cpu()))
                
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else: #stop if val loss starts increasing
                break
        opt.zero_grad()
        
    proj_layer.load_state_dict({"weight":best_weights})
    print("Best step:{}, Avg val similarity:{:.4f}".format(best_step, -best_val_loss.cpu()))
    
    #delete concepts that are not interpretable
    with torch.no_grad():
        outs = proj_layer(val_target_features.to(args.device).detach())
        sim = similarity_fn(val_concept_matrix.to(args.device).detach(), outs)
        interpretable = sim > args.interpretability_cutoff
        
    if args.print:
        for i, concept in enumerate(concepts):
            if sim[i]<=args.interpretability_cutoff:
                print("Deleting {}, Iterpretability:{:.3f}".format(concept, sim[i]))
    
    concepts = [concepts[i] for i in range(len(concepts)) if interpretable[i]]
    if len(concepts) == 0:
        raise ValueError("No concepts survived interpretability cutoff: {}".format(args.interpretability_cutoff))
    
    del concept_matrix, val_concept_matrix
    
    W_c = proj_layer.weight[interpretable]
    proj_layer = torch.nn.Linear(in_features=target_features.shape[1], out_features=len(concepts), bias=False)
    proj_layer.load_state_dict({"weight":W_c})
    
    output_proj = None
    test_metrics = None

    if args.dataset == "audioset":
        if hf_streaming:
            raise ValueError(
                "AudioSet final multi-label stage requires non-streaming mode. "
                "Run without --audioset_streaming for classifier fitting."
            )

        def _collect_targets(split_name, subset_name, expected_rows):
            ds = data_utils.get_audio_dataset(
                dataset_name=args.dataset,
                split=split_name,
                hf_streaming=False,
                hf_cache_dir=hf_cache_dir,
                max_items=hf_max_items,
                hf_subset=subset_name,
            )

            targets = []
            if hasattr(ds, "ds"):
                for row in ds.ds:
                    targets.append(_labels_to_multihot_tensor(row.get("labels", []), len(classes)))
                    if len(targets) >= expected_rows:
                        break
            else:
                for sample in ds:
                    target = sample["target"]
                    if not isinstance(target, torch.Tensor):
                        target = torch.as_tensor(target)
                    targets.append(target.float().cpu())
                    if len(targets) >= expected_rows:
                        break

            if len(targets) != expected_rows:
                raise ValueError(
                    "{} targets/features mismatch: {} vs {}".format(
                        split_name, len(targets), expected_rows
                    )
                )
            return torch.stack(targets, dim=0)

        train_targets = _collect_targets(d_train, hf_train_subset, target_features.shape[0])
        val_targets = _collect_targets(d_val, hf_val_subset, val_target_features.shape[0])
        test_targets = None
        if d_test is not None:
            test_targets = _collect_targets(d_test, hf_test_subset, test_target_features.shape[0])

        with torch.no_grad():
            train_c = proj_layer(target_features.detach())
            val_c = proj_layer(val_target_features.detach())

            train_mean = torch.mean(train_c, dim=0, keepdim=True)
            train_std = torch.std(train_c, dim=0, keepdim=True)
            train_std = torch.clamp(train_std, min=1e-6)

            train_c = (train_c - train_mean) / train_std
            val_c = (val_c - train_mean) / train_std

            train_y = train_targets.float()
            val_y = val_targets.float()

            test_c = None
            test_y = None
            if d_test is not None:
                test_c = proj_layer(test_target_features.detach())
                test_c = (test_c - train_mean) / train_std
                test_y = test_targets.float()

        train_ds = TensorDataset(train_c, train_y)
        train_loader = DataLoader(train_ds, batch_size=args.saga_batch_size, shuffle=True)

        linear = torch.nn.Linear(train_c.shape[1], len(classes)).to(args.device)
        linear.weight.data.zero_()
        linear.bias.data.zero_()

        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-3)

        best_val_loss = float("inf")
        best_step = 0
        best_state = {
            "weight": linear.weight.detach().clone(),
            "bias": linear.bias.detach().clone(),
        }
        train_iterator = iter(train_loader)

        for step in range(args.n_iters):
            try:
                x_batch, y_batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                x_batch, y_batch = next(train_iterator)

            x_batch = x_batch.to(args.device)
            y_batch = y_batch.to(args.device)

            optimizer.zero_grad()
            logits = linear(x_batch)
            bce_loss = criterion(logits, y_batch)
            reg_loss = args.lam * torch.mean(torch.abs(linear.weight))
            loss = bce_loss + reg_loss
            loss.backward()
            optimizer.step()

            if step % 50 == 0 or step == args.n_iters - 1:
                with torch.no_grad():
                    val_logits = linear(val_c.to(args.device))
                    val_loss = criterion(val_logits, val_y.to(args.device)) + args.lam * torch.mean(torch.abs(linear.weight))

                print(
                    "Step:{}, Train BCE:{:.4f}, Val BCE:{:.4f}".format(
                        step,
                        float(bce_loss.detach().cpu()),
                        float(val_loss.detach().cpu()),
                    )
                )

                if float(val_loss.detach().cpu()) < best_val_loss:
                    best_val_loss = float(val_loss.detach().cpu())
                    best_step = step
                    best_state = {
                        "weight": linear.weight.detach().clone(),
                        "bias": linear.bias.detach().clone(),
                    }
                elif step - best_step >= 400:
                    break

        linear.load_state_dict(best_state)
        W_g = linear.weight.detach().cpu()
        b_g = linear.bias.detach().cpu()
        output_proj = {
            "mode": "bce_multilabel",
            "best_step": int(best_step),
            "best_val_bce": float(best_val_loss),
            "lam": float(args.lam),
        }

        if d_test is not None:
            with torch.no_grad():
                logits_test = linear(test_c.to(args.device)).cpu()
                test_loss = criterion(logits_test, test_y).item()
                test_metrics = {"loss": test_loss}
                test_metrics.update(_compute_multilabel_metrics(logits_test, test_y, threshold=0.5))

            printable = ["loss: {:.4f}".format(test_metrics["loss"])]
            if test_metrics.get("mAP_macro") is not None:
                printable.append("mAP: {:.4f}".format(test_metrics["mAP_macro"]))
            if test_metrics.get("auc_roc_macro") is not None:
                printable.append("auc: {:.4f}".format(test_metrics["auc_roc_macro"]))
            if test_metrics.get("f1_micro") is not None:
                printable.append("f1_micro: {:.4f}".format(test_metrics["f1_micro"]))
            if test_metrics.get("f1_macro") is not None:
                printable.append("f1_macro: {:.4f}".format(test_metrics["f1_macro"]))
            print("Held-out test -> {}".format(", ".join(printable)))

    elif args.dataset in {"esc50", "urbansound8k"}:
        train_targets = [sample["label_idx"] for sample in data_utils.get_audio_dataset(args.dataset, d_train).samples]
        val_targets = [sample["label_idx"] for sample in data_utils.get_audio_dataset(args.dataset, d_val).samples]
        test_targets = None
        if d_test is not None:
            test_targets = [sample["label_idx"] for sample in data_utils.get_audio_dataset(args.dataset, d_test).samples]

        if len(train_targets) != target_features.shape[0]:
            raise ValueError("Train targets/features mismatch: {} vs {}".format(len(train_targets), target_features.shape[0]))
        if len(val_targets) != val_target_features.shape[0]:
            raise ValueError("Val targets/features mismatch: {} vs {}".format(len(val_targets), val_target_features.shape[0]))
        if d_test is not None and len(test_targets) != test_target_features.shape[0]:
            raise ValueError("Test targets/features mismatch: {} vs {}".format(len(test_targets), test_target_features.shape[0]))

        with torch.no_grad():
            train_c = proj_layer(target_features.detach())
            val_c = proj_layer(val_target_features.detach())

            train_mean = torch.mean(train_c, dim=0, keepdim=True)
            train_std = torch.std(train_c, dim=0, keepdim=True)
            train_std = torch.clamp(train_std, min=1e-6)

            train_c -= train_mean
            train_c /= train_std

            train_y = torch.LongTensor(train_targets)
            indexed_train_ds = IndexedTensorDataset(train_c, train_y)

            val_c -= train_mean
            val_c /= train_std

            val_y = torch.LongTensor(val_targets)
            val_ds = TensorDataset(val_c, val_y)

            if d_test is not None:
                test_c = proj_layer(test_target_features.detach())
                test_c -= train_mean
                test_c /= train_std
                test_y = torch.LongTensor(test_targets)

        indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)

        # Make linear model and zero initialize
        linear = torch.nn.Linear(train_c.shape[1],len(classes)).to(args.device)
        linear.weight.data.zero_()
        linear.bias.data.zero_()

        STEP_SIZE = 0.1
        ALPHA = 0.99
        metadata = {}
        metadata['max_reg'] = {}
        metadata['max_reg']['nongrouped'] = args.lam

        # Solve the GLM path
        output_proj = glm_saga(linear, indexed_train_loader, STEP_SIZE, args.n_iters, ALPHA, epsilon=1, k=1,
                          val_loader=val_loader, do_zero=False, metadata=metadata, n_ex=len(target_features), n_classes = len(classes))
        W_g = output_proj['path'][0]['weight']
        b_g = output_proj['path'][0]['bias']

        if d_test is not None:
            with torch.no_grad():
                logits_test = F.linear(test_c.to(W_g.device), W_g, b_g).cpu()
                test_loss = F.cross_entropy(logits_test, test_y).item()
                test_acc = (torch.argmax(logits_test, dim=1) == test_y).float().mean().item()
                test_metrics = {"loss": test_loss, "accuracy": test_acc}
            print("Held-out test -> loss: {:.4f}, acc: {:.4f}".format(test_metrics["loss"], test_metrics["accuracy"]))
    
    save_name = "{}/{}_cbm_{}".format(args.save_dir, args.dataset, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    os.mkdir(save_name)
    torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
    torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
    torch.save(W_c, os.path.join(save_name ,"W_c.pt"))
    torch.save(W_g, os.path.join(save_name, "W_g.pt"))
    torch.save(b_g, os.path.join(save_name, "b_g.pt"))
    
    with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write('\n'+concept)
    
    with open(os.path.join(save_name, "args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
        out_dict = {}
        if args.dataset in {"esc50", "urbansound8k"}:
            for key in ('lam', 'lr', 'alpha', 'time'):
                out_dict[key] = float(output_proj['path'][0][key])
            out_dict['metrics'] = output_proj['path'][0]['metrics']
        else:
            out_dict['training'] = output_proj
        out_dict['splits'] = {"train": d_train, "val": d_val, "test": d_test}
        if args.dataset == "audioset":
            out_dict['subsets'] = {
                "train": hf_train_subset,
                "val": hf_val_subset,
                "test": hf_test_subset,
            }
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
        if test_metrics is not None:
            out_dict['test_metrics'] = test_metrics
        json.dump(out_dict, f, indent=2)
    
if __name__=='__main__':
    args = parser.parse_args()
    train_cbm_and_save(args)