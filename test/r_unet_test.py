import torch

from src.surgical_copilot.models.Recurrent_U_Net.runet import RecurrentUNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =========================
    # 1. Config modello
    # =========================
    model = RecurrentUNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=[32, 64, 128, 256, 512],
        strides=[2, 2, 2, 2],
        num_res_units=2,
        recurrent_type="gru",
        recurrent_layers=1,
    ).to(device)

    model.eval()

    # =========================
    # 2. Fake input sequence
    # =========================
    B = 2
    T = 5
    C = 3
    H = 256
    W = 256

    x = torch.randn(B, T, C, H, W).to(device)

    # =========================
    # 3. Forward pass
    # =========================
    with torch.no_grad():
        y = model(x).to(device)
        print(T, y.shape)


        # =========================
        # 4. Check output
        # =========================
        print("\n===== OUTPUT SHAPE =====")
        print("Input :", x.shape)
        print("Output:", y.shape)

        # sanity check
        assert y.shape[0] == B
        assert y.shape[1] == T

    print("\nOK: forward pass completato correttamente.")


if __name__ == "__main__":
    main()