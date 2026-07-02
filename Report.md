# Report

## Spatial Model baseline
### Sintesi Prestazioni Modelli
Questa tabella espone i risultati sintetici delle performance dei modelli considerati spaziali.

| Modello | Fold Usati | Dice Medio | Dice Std | IoU Medio | IoU Std | HD95 Medio | HD95 Std | FPS Medio | FPS Std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unet | 5 | 0.613 | 0.042 | 0.500 | 0.033 | 143.4 | 19.9 | 74.4 | 0.5 |
| unet_resnet18 | 5 | 0.607 | 0.037 | 0.493 | 0.031 | 148.3 | 10.1 | 60.4 | 4.3 |
| unet_plus_plus | 3 | 0.655 | 0.016 | 0.535 | 0.013 | 133.2 | 7.4 | 31.7 | 0.5 |
| unet_plus_plus_resnet18 | 5 | 0.620 | 0.068 | 0.501 | 0.059 | 151.3 | 8.6 | 48.2 | 0.3 |
| swin_unetr | 5 | 0.608 | 0.057 | 0.495 | 0.043 | 149.9 | 17.7 | 17.8 | 0.2 |
| yolo_v8_seg | 5 | 0.599 | 0.038 | 0.483 | 0.032 | 145.2 | 9.2 | 46.1 | 1.0 |

### Metriche per fold

| Fold | Modello | Dice | IoU | HD95 |
| --- | --- | --- | --- | --- |
| **Fold 0** | unet | 0.655 | 0.536 | 129.5 |
|  | unet_resnet18 | 0.627 | 0.509 | 146.5 |
|  | unet_plus_plus | 0.650 | 0.532 | 134.9 |
|  | unet_plus_plus_resnet18 | 0.686 | 0.560 | 142.1 |
|  | swin_unetr | 0.665 | 0.536 | 134.6 |
|  | yolo_v8_seg | 0.643 | 0.519 | 142.5 |
| **Fold 1** | unet | 0.605 | 0.490 | 148.9 |
|  | unet_resnet18 | 0.618 | 0.504 | 134.1 |
|  | unet_plus_plus | 0.673 | 0.550 | 124.9 |
|  | unet_plus_plus_resnet18 | 0.637 | 0.517 | 155.0 |
|  | swin_unetr | 0.598 | 0.495 | 149.4 |
|  | yolo_v8_seg | 0.603 | 0.490 | 148.2 |
| **Fold 2** | unet | 0.638 | 0.523 | 123.6 |
|  | unet_resnet18 | 0.596 | 0.485 | 154.2 |
|  | unet_plus_plus | 0.642 | 0.523 | 139.8 |
|  | unet_plus_plus_resnet18 | 0.622 | 0.501 | 154.0 |
|  | swin_unetr | 0.603 | 0.490 | 158.3 |
|  | yolo_v8_seg | 0.595 | 0.482 | 147.4 |
| **Fold 3** | unet | 0.569 | 0.463 | 165.4 |
|  | unet_resnet18 | 0.551 | 0.450 | 167.6 |
|  | unet_plus_plus_resnet18 | 0.532 | 0.435 | 169.5 |
|  | swin_unetr | 0.547 | 0.447 | 170.8 |
|  | yolo_v8_seg | 0.539 | 0.436 | 157.0 |
| **Fold 4** | unet | 0.597 | 0.489 | 149.6 |
|  | unet_resnet18 | 0.643 | 0.518 | 139.1 |
|  | unet_plus_plus_resnet18 | 0.623 | 0.492 | 136.0 |
|  | swin_unetr | 0.629 | 0.508 | 136.5 |
|  | yolo_v8_seg | 0.613 | 0.487 | 130.8 |


### U-Net baseline senza ResNet

#### Metriche di valutazione baseline

| Metrica | Media ± std | Min–max |
|---|---:|---:|
| Dice | **0.613 ± 0.042** | 0.554–0.645 |
| IoU | **0.500 ± 0.033** | 0.454–0.527 |
| HD95 | **143.41 ± 19.86** | 124.14–172.16 |
| Validation loss | **0.299 ± 0.035** | 0.260–0.327 |
| Inference FPS | **74.39 ± 0.47** | 73.75–74.81 |

#### Perturbazioni / stress test

| Perturbazione | Dice ↑ | IoU ↑ | HD95 ↓ | Drop loggato ↓ |
|---|---:|---:|---:|---:|
| `intensity_shift_only` | **0.665 ± 0.098** | **0.544 ± 0.104** | **119.56 ± 32.92** | **-0.006 ± 0.013** |
| `specular_only` | **0.648 ± 0.100** | **0.527 ± 0.102** | **123.80 ± 31.16** | **0.020 ± 0.013** |
| `contrast_only` | **0.632 ± 0.095** | **0.514 ± 0.096** | **124.85 ± 35.74** | **0.044 ± 0.042** |
| `noise_only` | **0.574 ± 0.044** | **0.455 ± 0.050** | **148.16 ± 12.41** | **0.125 ± 0.073** |
| `smoke_only` | **0.514 ± 0.066** | **0.387 ± 0.050** | **163.43 ± 36.74** | **0.219 ± 0.061** |
| `blur_only` | **0.250 ± 0.114** | **0.213 ± 0.097** | **281.86 ± 41.88** | **0.602 ± 0.214** |
| `chirurgical_worst_case` | **0.104 ± 0.035** | **0.071 ± 0.027** | **320.62 ± 23.26** | **0.835 ± 0.075** |

- La performance baseline pulita è circa **Dice 0.613 / IoU 0.500** mentre per l'HD\95 è **143.41**. Significa che la U-Net è in grado di segmentare abbastanza bene le immagini pulite, con un'accuratezza media del 61.3% e un'area di sovrapposizione del 50%. Tuttavia, l'HD95 indica che ci sono ancora discrepanze significative tra le maschere predette e quelle reali.
- La U-Net è abbastanza robusta a **intensity shift**, **specular** e **contrast**, con Dice medio vicino o superiore al baseline.
- Le perturbazioni più distruttive sono nettamente:
  1. **`chirurgical_worst_case`**: Dice medio **0.104**, drop **0.835**
  2. **`blur_only`**: Dice medio **0.250**, drop **0.602**
- `smoke_only` e `noise_only` degradano ma non collassano: Dice medio rispettivamente **0.514** e **0.574**.

### Unet-ResNet18 baseline

#### Metriche baseline

| Metrica | Media ± std | Min–max |
|---|---:|---:|
| Dice | **0.607 ± 0.037** | 0.562–0.651 |
| IoU | **0.493 ± 0.031** | 0.456–0.531 |
| HD95 | **148.31 ± 10.06** | 134.78–162.21 |
| Validation loss | **0.207 ± 0.022** | 0.188–0.244 |
| Inference FPS | **60.39 ± 4.29** | 55.20–66.54 |

#### Perturbazioni / stress test

| Perturbazione | Dice ↑ | IoU ↑ | HD95 ↓ | Drop loggato ↓ |
|---|---:|---:|---:|---:|
| `specular_only` | **0.668 ± 0.088** | **0.540 ± 0.097** | **121.65 ± 31.16** | **0.010 ± 0.014** |
| `intensity_shift_only` | **0.665 ± 0.093** | **0.538 ± 0.100** | **121.61 ± 30.71** | **0.015 ± 0.008** |
| `contrast_only` | **0.655 ± 0.080** | **0.525 ± 0.086** | **111.07 ± 27.06** | **0.028 ± 0.026** |
| `smoke_only` | **0.586 ± 0.080** | **0.453 ± 0.084** | **143.59 ± 28.68** | **0.132 ± 0.055** |
| `noise_only` | **0.505 ± 0.090** | **0.384 ± 0.087** | **153.12 ± 23.68** | **0.250 ± 0.102** |
| `blur_only` | **0.003 ± 0.007** | **0.002 ± 0.004** | **93.42 ± 129.69** | **0.996 ± 0.008** |
| `chirurgical_worst_case` | **0.001 ± 0.001** | **0.000 ± 0.001** | **205.03 ± 188.09** | **0.999 ± 0.001** |


- La **UNet-ResNet18 baseline recente** ha performance pulita molto simile alla UNet senza ResNet:  
  **Dice 0.607 vs 0.613**, **IoU 0.493 vs 0.500**.
- È robusta a **specular**, **intensity shift** e **contrast**: queste perturbazioni superano il Dice baseline medio.
- Collassa quasi completamente su:
  - **`blur_only`**: Dice medio **0.003**, drop **0.996**
  - **`chirurgical_worst_case`**: Dice medio **0.001**, drop **0.999**
- Nota: per `blur_only` e in parte `chirurgical_worst_case`, l’HD95 contiene valori `0.0` in alcuni fold; quindi per queste due perturbazioni l’HD95 medio è poco interpretabile rispetto a Dice/IoU/drop.
