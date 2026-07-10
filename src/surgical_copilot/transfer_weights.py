import torch
import os

def transfer_weights_to_temporal(baseline_path: str, save_path: str, target_layer_name: str, new_channels: int = 4):
    """
    Universal function to transfer weights from an RGB network to a temporal network.
    """
    if not os.path.exists(baseline_path):
        print(f"[!] ERRORE: File baseline non trovato in {baseline_path}")
        return False

    print(f"[*] -> Estrazione pesi da: {baseline_path}")

    baseline_state_dict = torch.load(baseline_path, map_location="cpu")
    temporal_state_dict = {}

    for layer_name, weights in baseline_state_dict.items():
        # Se il layer è una convoluzione che accetta input (es. ha 3 canali)
        if weights.ndim == 4 and weights.shape[1] == 3:
            out_ch, _, kh, kw = weights.shape
            new_weights = torch.zeros((out_ch, new_channels, kh, kw), dtype=weights.dtype)
            # Copia i canali RGB esistenti
            new_weights[:, :3, :, :] = weights.clone()
            # Inizializza il 4° canale (es. media dei canali RGB per mantenere il range)
            new_weights[:, 3, :, :] = weights.mean(dim=1)
            temporal_state_dict[layer_name] = new_weights
            print(f"[*] Layer trovato: {layer_name}, shape originale: {weights.shape}")
        else:
            # Per tutti gli altri layer (bias, batchnorm, residuali 1x1, ecc.)
            temporal_state_dict[layer_name] = weights.clone()

    torch.save(temporal_state_dict, save_path)
    print(f"[*] -> Pesi temporali salvati in: {save_path}")
    return True

def load_or_create_temporal_weights(model, fold_idx: int, device, target_layer_name: str, pretrained_weights_path: str):
    """
    Manage the loading or creation of temporal weights for a given model and fold index.
    If the temporal weights do not exist, it will attempt to create them from the baseline weights.
    If the baseline weights are not found, it will start training from scratch.
    """

    if pretrained_weights_path is None or str(pretrained_weights_path).lower() == "none":
        print(f"[!] pretrained_weights_path non fornito per il Fold {fold_idx}: partenza da zero.")
        return model

    nome_classe_modello = model.__class__.__name__ 
    path_baseline = f"/work/cvcs2026/DeepLook/results/weights/{nome_classe_modello}/best_fold{fold_idx}.pth"
    path_temporal = f"/work/cvcs2026/DeepLook/results/weights/{nome_classe_modello}/best_fold{fold_idx}_4ch.pth"
    
    # Crea i pesi temporali se non esistono
    if not os.path.exists(path_temporal):
        if os.path.exists(path_baseline):
            print(f"\n[*] Generazione pesi temporali per Fold {fold_idx} in corso...")
            success = transfer_weights_to_temporal(
                baseline_path=path_baseline,
                save_path=path_temporal,
                target_layer_name=target_layer_name,
                new_channels=4
            )
            if not success:
                print(f"[!] Fallimento nella generazione dei pesi per il Fold {fold_idx}.")
        else:
            print(f"\n[!] Impossibile generare pesi temporali: baseline {path_baseline} non trovata.")

    # Carica i pesi nel modello
    if os.path.exists(path_temporal):
        print(f"[*] CARICAMENTO PESI PRE-ADDESTRATI TEMPORALI: {path_temporal}")
        model.load_state_dict(torch.load(path_temporal, map_location=device), strict=False)
    else:
        print(f"[!] PARTENZA DA ZERO per il Fold {fold_idx} temporale.")
        
    return model