import hydra
import torch
import gc
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from surgical_copilot.bench.BenchmarkEngine import BenchmarkEngine
from surgical_copilot.HemoDataset import HemosetDataSet
from surgical_copilot.bench.perturbation import PerturbationPipelines
from surgical_copilot.transfer_weights import load_or_create_temporal_weights

@hydra.main(version_base=None, config_path="../../../configs", config_name="config") 
def benchmarking(cfg: DictConfig):
    print("Script avviato!")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[*] Device selezionato: {device}")

    # Dataset Loading
    print(f"[*] Loading dataset") 
    dataset = HemosetDataSet(root_dir=cfg.data.root_dir,seed=cfg.seed, image_size=cfg.data.img_size)
    
    all_fold_metrics = []
    model_key = cfg.model_key

    for fold in range(cfg.data.n_folds):
        print(f"\n{'#'*50}\n[*] Fold {fold+1}/{cfg.data.n_folds}\n{'#'*50}")
       
        # ESTRAZIONE DINAMICA DA HYDRA 
        # Convertiamo la configurazione di Hydra del modello in un dizionario Python
        model_cfg = OmegaConf.to_container(cfg.model[model_key], resolve=True)
        
        # Estraiamo e RIMUOVIAMO is_temporal dal dizionario (di default False se non specificato).
        # In questo modo Hydra non andrà in crash cercando di passarlo all' __init__ del modello.
        is_temporal_model = model_cfg.pop("is_temporal", False)
        target_layer = model_cfg.pop("temporal_target_layer", None)

        if is_temporal_model:
            if target_layer is None:
                print(f"[!] ERRORE FATALE: Hai impostato is_temporal=True ma manca 'temporal_target_layer' nel models.yaml per {model_key}!")
                exit(1)
        
        # CONFIGURAZIONE DATALOADER ---
        # Per i modelli temporali autoregressivi, il batch_size DEVE essere 1
        b_size = 1 if is_temporal_model else OmegaConf.select(cfg, "data.batch_size", default=4)
        num_w = OmegaConf.select(cfg, "data.num_workers", default=4)
        
        # same loader for each model
        train_loader, val_loader, test_loader = dataset.get_loaders(
            fold_idx=fold,
            n_splits=cfg.data.n_folds,
            batch_size=b_size,                
            num_workers=num_w,
            train_transforms=PerturbationPipelines.get_train_pipeline(), 
            temporal_mode=is_temporal_model   
        )
        print(f"\n{'='*30}\n[*] BENCHMARKING MODEL: {model_key}\n{'='*30}")

        # ISTANZIAZIONE DEL MODELLO 
        # Attenzione: Usiamo 'model_cfg' (il dizionario pulito) 
        model = instantiate(model_cfg).to(device)
        
        # istanziazione del modello temporale
        if is_temporal_model:
            # Passiamo la variabile estratta da Hydra e il fold corrente
            model = load_or_create_temporal_weights(
                model=model, 
                fold_idx=fold, 
                device=device,
                target_layer_name=target_layer  
            )

        # training components
        optimizer = instantiate(cfg.trainer.optimizer, params=model.parameters())
        #scheduler = instantiate(cfg.trainer.scheduler, optimizer=optimizer)

        warmup_epochs = 5
        max_epochs = cfg.trainer.trainer.max_epochs

        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.01, 
            total_iters=warmup_epochs
        )

        # Fase 2: Cosine Annealing per le epoche rimanenti
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=max_epochs - warmup_epochs, 
            eta_min=1e-6
        )

        # Fase 3: Concatenazione
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_epochs]
        )

        loss_fn = instantiate(cfg.trainer.loss)
        scaler = instantiate(cfg.trainer.scaler)

        # training engine
        engine = BenchmarkEngine(
            model=model, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            test_loader=test_loader,
            optimizer=optimizer, 
            scheduler=scheduler, 
            loss_fn=loss_fn, 
            scaler=scaler, 
            cfg=cfg, 
            device=device,
            is_temporal=is_temporal_model  
        )

        if cfg.logging.wandb_enabled:
            resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
            exp_name = cfg.logging.get("exp_tag", "baseline")
            wandb.init(
                project=cfg.logging.project, 
                config=resolved_cfg, 
                group=f"{exp_name}_fold_{fold+1}", 
                name=f"{model_key}_fold_{fold}",
                reinit=True,
                tags=exp_name

            )

        fold_metrics = engine.run()
        all_fold_metrics.append(fold_metrics)

        if cfg.logging.wandb_enabled:
            wandb.finish()

        del model, optimizer, scheduler, engine
        torch.cuda.empty_cache()
        gc.collect()
    
    dice_list = [x["dice"] for x in all_fold_metrics]
    hd95_list = [x["hd95"] for x in all_fold_metrics]

    mean_dice, std_dice = np.mean(dice_list), np.std(dice_list)
    mean_hd95, std_hd95 = np.mean(hd95_list), np.std(hd95_list)

    if cfg.logging.wandb_enabled:
        wandb.init(
            project=cfg.logging.project, 
            group=model_key, 
            job_type="final_stats", 
            name=f"{model_key}_summary",
        )
        wandb.log({
            "mean_dice": mean_dice,
            "std_dice": std_dice,
            "mean_hd95": mean_hd95,
            "std_hd95": std_hd95
        })
        wandb.finish()

    print("\n" + "=" * 80)
    print(f"FINAL RESULTS | Dice: {mean_dice:.4f} ± {std_dice:.4f} | HD95: {mean_hd95:.4f} ± {std_hd95:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    benchmarking()