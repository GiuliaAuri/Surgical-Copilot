# Surgical_copilot

Author: cla&giu
Python: 3.11

## Configurazioni dell'esperimento

### 1. Dataset e Data Management

* **Dataset Target**: Il sistema si basa su `HemosetDataSet`, il quale gestisce il caricamento delle immagini chirurgiche e delle relative maschere di annotazione per la segmentazione ematica.
* **Validazione**: È implementato un protocollo di validazione incrociata K-Fold gestito dinamicamente tramite configurazione (`cfg.data.n_folds`) per garantire la robustezza statistica delle metriche.
* **Pipeline di Perturbazione**: I data loader integrano pipeline di data augmentation e stress test avanzati tramite `PerturbationPipelines.get_train_pipeline()`, necessarie per testare la robustezza dei modelli sotto condizioni avverse quali fumo, rumore e riflessi speculari.

### 2. Configurazione e Pipeline di Training

**Training Loop & Dataloading**

* **Gradient Accumulation:** Mantenendo un batch size fisico di 4 (necessario per non saturare la VRAM, specialmente con modelli pesanti o input ad alta risoluzione), accumuli i gradienti per 3 step prima di aggiornare i pesi. Questo genera un *effective batch size* di 12, garantendo stime del gradiente molto più stabili senza richiedere hardware di fascia superiore.
* **Scheduling delle Valutazioni:** L'evaluation sul validation set è fissata ogni 5 epoche (`eval_freq: 5`). Questo riduce il tempo complessivo di run, evitando di bloccare inutilmente le GPU per calcolare le metriche di validazione ad ogni singola epoca. Il limite massimo è impostato a 50 epoche.
* **Dataloading:** Impostare `num_workers: 4` assicura che la CPU prepari e trasferisca i tensori in parallelo senza creare colli di bottiglia verso le code della GPU.

**Loss Function**

* Viene usata la `DiceFocalLoss` (combinazione lineare 50/50 tra Dice e Focal Loss) rispetto alla `Cross-Entropy multi-classe` usata come baseline nel paper di HemoSet che risulta meno efficace per la segmentazione di oggetti piccoli e sbilanciati come il sangue.
    * La componente **Dice** massimizza l'overlap geometrico mitigando lo sbilanciamento di classe, 
    * **Focal Loss** ($\gamma = 1.0$) penalizza dinamicamente i pixel classificati con bassa confidenza, costringendo il network a concentrarsi sui bordi difficili (es. riflessi speculari o fumo) e ignorando i veri negativi "facili" del background. 
Il parametro `sigmoid: True` serve ad appiattire  l'output del modello e trattarlo come classificazione binaria per ogni pixel.

**Ottimizzazione e Precisione Mista**

* **Ottimizzatore:** L'uso di `AdamW` (Adam con Weight Decay disaccoppiato a $10^{-5}$) associato a un learning rate di $10^{-3}$ è lo standard de facto per i modelli di visione moderni, offrendo una convergenza veloce ma fortemente regolarizzata.
* **Mixed Precision (AMP):** Il parametro `precision: "16-mixed"` accoppiato al `GradScaler` abilitato è fondamentale. Il calcolo in FP16 dimezza i consumi di memoria e raddoppia quasi il throughput computazionale (sfruttando i Tensor Cores dell'hardware), mentre lo scaler previene l'underflow dei gradienti (cioè valori che diventerebbero zero nella precisione a 16-bit) mantenendo la stabilità dell'aggiornamento.
* **Scheduler** Viene usato un `SequentialLR` a due fasi (Warmup lineare + Cosine Annealing) che permette di iniziare l'addestramento con un learning rate basso per i primi 5 step (warmup) e poi decrescerlo secondo una curva coseno, riducendo il rischio di oscillazioni e migliorando la convergenza finale.

**Regolarizzazione**

* **Early Stopping**: L'addestramento verrà interrotto anticipatamente se il `val_dice` non migliora per 5 cicli di valutazione consecutivi (`patience: 5`). Poiché valutiamo ogni 5 epoche, una pazienza di 5 significa che l'Early Stopping interverrà dopo **25 epoche** senza miglioramenti sul set di validazione.

### 3. Tassonomia dei Modelli Implementati

1. **Modelli Spaziali Pre-addestrati**: Classi `SMPUNet` e `SMPUNetPlusPlus` configurate esplicitamente con encoder ResNet18 e pesi pre-addestrati su ImageNet, rispecchiando la metodologia di inizializzazione descrittiva delle baseline del dataset HemoSet. Abbiamo, quindi esteso le classi di MONAI per includere questi modelli SMP, garantendo compatibilità con il framework di addestramento e validazione.
2. **Modelli Temporali / Ricorrenti**: Architetture dedicate all'analisi della coerenza sequenziale (Early Fusion e Late Fusion). Spicca la classe `RecurrentSMPUNet` che combina l'encoder spaziale ResNet18 pre-addestrato con un modulo ricorrente (GRU o LSTM) applicato direttamente al livello del bottleneck. Per questi modelli è previsto il congelamento iniziale del backbone (`freeze_backbone=True`) e una fase di warm-start.

### 4. Protocollo di Valutazione e Metriche

Il calcolo prestazionale a fine esecuzione estrae le seguenti metriche di accuratezza spaziale e geometrica:

* **Dice Score** (F1-score della segmentazione pixel-level).
* **Intersection over Union (IoU)**.
* **Hausdorff Distance 95 (HD95)** per valutare l'errore di distanza sui contorni delle aree di accumulo ematico.
* **Metriche Temporali**: Il modulo `InterFrameTemporalMetric` è predisposto per misurare la stabilità inter-frame tramite metriche di temporal IoU e temporal Dice, quantificando la consistenza delle predizioni nel tempo.

## Risultati

### Baseline

### Early Fusion

con e senza warm-start.

### Late Fusion

con e senza warm-start.


