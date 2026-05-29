import torch
import os

def transfer_weights_to_temporal(baseline_path: str, save_path: str, target_layer_name: str, new_channels: int = 4):
    """
    Funzione universale per trasferire i pesi da una rete RGB a una rete temporale.
    """
    if not os.path.exists(baseline_path):
        print(f"[!] ERRORE: File baseline non trovato in {baseline_path}")
        return False

    print(f"[*] -> Estrazione pesi da: {baseline_path}")
    baseline_state_dict = torch.load(baseline_path, map_location="cpu")
    temporal_state_dict = {}
    layer_found = False

    for layer_name, weights in baseline_state_dict.items():
        if layer_name == target_layer_name:
            layer_found = True
            out_ch, in_ch, kh, kw = weights.shape
            
            # Crea tensore vuoto a 4 canali
            new_weights = torch.zeros((out_ch, new_channels, kh, kw), dtype=weights.dtype)
            
            # Copia i canali disponibili (RGB)
            channels_to_copy = min(in_ch, new_channels)
            new_weights[:, :channels_to_copy, :, :] = weights[:, :channels_to_copy, :, :].clone()
            
            temporal_state_dict[layer_name] = new_weights
        else:
            temporal_state_dict[layer_name] = weights.clone()

    if not layer_found:
        print(f"[!] ATTENZIONE: Il layer '{target_layer_name}' non esiste in questa rete!")
        return False

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(temporal_state_dict, save_path)
    print(f"[*] -> Pesi temporali creati con successo in: {save_path}")
    return True

def load_or_create_temporal_weights(model, fold_idx: int, device, target_layer_name: str):
    """
    Gestisce automaticamente la ricerca, la creazione e il caricamento 
    dei pesi per una rete temporale, rispettando il Fold corrente.
    """
    nome_classe_modello = model.__class__.__name__ 
    path_baseline = f"results/best_{nome_classe_modello}_fold{fold_idx}.pth"
    path_temporal = f"results/temporal_pretrained_fold{fold_idx}.pth"
    
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