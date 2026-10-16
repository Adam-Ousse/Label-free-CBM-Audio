AST backbone integration notes

- file: `models/ast_backbone.py`
- default alias: `ast_audioset` -> `MIT/ast-finetuned-audioset-10-10-0.4593`
- esc50 alias: `ast_esc50` -> `Adam-ousse/ast-esc50-finetuned-fold1`
- output contract: pooled embedding tensor with shape `[batch, 768]`
- accepted input shapes: `[batch, time]` or `[batch, 1, time]`

note

- `ASTAudioBackbone` uses `ASTModel`, not the classifier head.
- use `models/ast_classifier.py` for logits/accuracy evaluation with `ASTForAudioClassification`.

custom model ids

- use backbone name format: `ast_hf__ORG__REPO`
- example: `ast_hf__MIT__ast-finetuned-audioset-10-10-0.4593`

where it is used

- `data_utils.get_target_model()`
- `utils.save_backbone_audio_features()`
- `cbm.py` backbone construction