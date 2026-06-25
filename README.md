# Surgical_copilot

Author: cla&giu
Python: 3.11

## Configurazioni dell'esperimento

### 1. Dataset e Data Management

* **Dataset Target**: Il sistema si basa su `HemosetDataSet`, il quale gestisce il caricamento delle immagini chirurgiche e delle relative maschere di annotazione per la segmentazione ematica.
* **Validazione**: È implementato un protocollo di validazione incrociata K-Fold gestito dinamicamente tramite configurazione (`cfg.data.n_folds`) per garantire la robustezza statistica delle metriche.
* **Pipeline di Perturbazione**: I data loader integrano pipeline di data augmentation e stress test avanzati tramite `PerturbationPipelines.get_train_pipeline()`, necessarie per testare la robustezza dei modelli sotto condizioni avverse quali fumo, rumore e riflessi speculari.
* **Strategia di Split (Group K-Fold) e Dataloaders**: L'uso di GroupKFold(n_splits=n_splits) di scikit-learn garantisce che i frame appartenenti allo stesso paziente ("pig") finiscano tutti nello stesso fold (train, val o test).
Se si facesse uno split randomico sui frame (KFold classico), un frame quasi identico (perché distanziato di pochi millisecondi) potrebbe finire nel train e il successivo nel test, inficiando completamente il senso della valutazione, prevenedo così Data Leakage (Patient-Level Split)
* **Partizionamento dei Set (Train, Val, Test)**: Per ogni fold, il set di validazione viene ritagliato prendendo il 25% finale della lista mista dei pazienti destinati al training. 
* **Gestione del Dataloading Spaziale vs. Temporale**:
   * **Modelli Spaziali** (temporal_mode=False): La lista globale dei frame di training subisce uno shuffling (self.rng.shuffle(train_files)) prima di creare il dataset, ed è attivo lo shuffling anche nel DataLoader PyTorch. Questo garantisce l'estrazione I.I.D. (Independent and Identically Distributed) classica necessaria alle U-Net spaziali.
  * **Modelli Temporali** (temporal_mode=True): Lo shuffling viene inibito. Questo è obbligatorio. Per far sì che le recurrent neural networks (RNN, GRU, LSTM) apprendano le dinamiche temporali corrette, il dataloader deve fornire i frame nell'esatto ordine cronologico con cui sono stati registrati dall'endoscopio.

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

## Perturbation Engine

Questo modulo `PerturbationPipelines`permette di simulare sinteticamente ma in modo realistico le condizioni avverse tipiche di una sala operatoria.

### Training Pipeline: Data Augmentation Strategica

La funzione `get_train_pipeline()` non applica le perturbazioni estreme, ma si concentra su una *data augmentation* classica per migliorare la generalizzazione di base (evitando overfitting).

* **Campionamento Bilanciato (`RandCropByPosNegLabeld`)**: Questa è una scelta eccellente per task altamente sbilanciati come l'emorragia. Invece di ritagliare patch a caso, forza il dataloader a estrarre patch che contengono il target (sangue) con probabilità doppia (`pos=2`) rispetto al background (`neg=1`). Questo assicura che il modello veda abbastanza esempi di emorragia in ogni batch.
* **Trasformazioni Geometriche**: `RandFlipd` (specchiamento) e `RandRotated` (rotazione) insegnano al modello l'invarianza spaziale (un'emorragia mantiene la sua semantica indipendentemente dall'angolo della telecamera endoscopica).
* **Deformazioni Elastiche (`Rand2DElasticd`)**: Simula i micro-movimenti e le deformazioni dei tessuti molli (es. peristalsi o respirazione del paziente) durante l'intervento.
* **Variazioni di Contrasto (`RandAdjustContrastd`)**: Simula leggere variazioni di illuminazione nativa dell'endoscopio.

*Nota:* Questa configurazione estende quella degli autori di HemoSet.

### Evaluation Scenarios: Il Framework di Stress-Test

La funzione `get_eval_scenarios()` definisce un dizionario di pipeline. Durante la fase di inferenza/test, il modello verrà valutato iterativamente su ciascuno di questi scenari.

1. **`noise_only` (Sensore)**: Applica rumore Gaussiano (`RandGaussianNoised`). Simula il rumore termico o di lettura tipico dei sensori CMOS delle camere endoscopiche, specialmente in condizioni di bassa luminosità nelle cavità corporee.
2. **`blur_only` (Movimento)**: Usa `RandGaussianSmoothd`. Simula la sfocatura (motion blur) causata da movimenti rapidi del braccio robotico o da sfuocature (defocus) improvvise della lente.
3. **`intensity_shift_only` / `contrast_only` (Illuminazione)**: Modificano l'istogramma dell'immagine. Simulano le repentine variazioni di luminosità quando la fonte di luce si avvicina o si allontana dai tessuti, o quando cambia l'assorbimento della luce da parte degli organi.
4. **`smoke_only` (Condizione Chirurgica Primaria)**: Usa la tua classe custom `RandSurgicalSmoked`. Genera un pattern di rumore a bassa risoluzione (tipo Perlin/Simplex), lo interpola e lo usa come mappa di opacità (alpha blending). È una simulazione matematicamente elegante del fumo generato dall'elettrobisturi o dal laser durante la cauterizzazione.
5. **`specular_only` (Condizione Chirurgica Primaria)**: Usa `RandSpecularReflectiond`. I tessuti bagnati (e il sangue stesso) creano forti riflessi speculari sotto la luce coassiale dell'endoscopio. La classe genera "blob" localizzati di altissima intensità per saturare parzialmente il sensore (clipping dei bianchi), ingannando i filtri di segmentazione classici.
6. **`chirurgical_worst_case`**: La combinazione di *tutti* i precedenti. È il test di robustezza definitivo. Se la tua U-Net temporale sopravvive qui rispetto alla baseline, il claim del paper è blindato.

## Risultati

### Baseline

### Early Fusion

con e senza warm-start.

### Late Fusion

con e senza warm-start.


