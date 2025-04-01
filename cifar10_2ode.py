import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import math
import argparse
import wandb

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from harmonization_loss import standardize_cut, pyramidal_mse_with_tokens, loss_norm
from blur import gaussian_blur, gaussian_kernel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gaussian_kernel = torch.tensor(gaussian_kernel(), dtype=torch.float32)

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        n = 32
        self.conv1 = nn.Conv2d(3, n, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(n, n*2, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1_input_size = (n*2) * 16 * 16
        self.fc1 = nn.Linear(self.fc1_input_size, n*4)
        self.fc2 = nn.Linear(n*4, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def create_mask(mask, height, mask_size=16):
    mask = torch.zeros((mask.size(0), mask.size(2), mask.size(3))).to(device)
    start = (height - mask_size) // 2
    end = start + mask_size
    mask[:, start:end, start:end] = 1
    # mask = torch.mean(mask, dim=1)
    return mask

def create_composite_image(images):
    """
    指定されたインデックスの画像のみを残し、それをグリッド画像として出力する。

    Args:
        images (torch.Tensor): 画像バッチ [B, C, H, W]
    Returns:
        torch.Tensor: [N, 1, 32, 32] のグリッド画像（N は mask_indices_list の要素数）
    """
    output_list = []
    true_heatmaps_list = []
    mix_num = images.size(0)
    if images.size(0) < wandb.config["batch_size"]:
        tmp_image = torch.zeros((wandb.config["batch_size"], images.size(1), images.size(2), images.size(3)), dtype=torch.float32, device=images.device)
        tmp_image[:images.size(0)] = images
        images = tmp_image

    true_heatmaps = torch.zeros_like(images, dtype=torch.float32, device=images.device)
    true_heatmaps[4] = 1.0  # maskの部分を1にする
    for mask_indices in range(mix_num):
        composite_batch = images.clone()
        # mask = torch.zeros(images.size(0), dtype=torch.bool, device=images.device)
        # mask[mask_indices] = True  # 指定したインデックスのみ True にする
        # composite_batch[~mask] = 0  # 指定していないインデックスを 0 にマスク

        # グリッドにまとめる（例：√B × √Bのレイアウト）
        nrow = int(math.sqrt(images.size(0)))
        if mask_indices != 4:
            # composite_batch[0] と composite_batch[4] を入れ替える
            # composite_batch_0_copy = composite_batch[mask_indices].clone()  # 0番目をコピー
            # composite_batch_4_copy = composite_batch[4].clone()  # 4番目をコピー
            # コピーを入れ替える
            composite_batch[mask_indices] = images[4]
            composite_batch[4] = images[mask_indices]


        grid = torchvision.utils.make_grid(composite_batch, nrow=nrow, padding=0, normalize=True, scale_each=False, pad_value=0)
        grid_true_heatmaps = torchvision.utils.make_grid(true_heatmaps, nrow=nrow, padding=0, normalize=True, scale_each=False, pad_value=0)
        # チャンネル数が 3 になった場合、C=1 に戻す
        if grid.shape[0] != composite_batch.shape[1]:
            grid = grid.mean(dim=0, keepdim=True)  #     [1, H_grid, W_grid]
        if grid_true_heatmaps.shape[0] != 1:
            # grid_true_heatmaps = grid_true_heatmaps.mean(dim=0, keepdim=True)
            grid_true_heatmaps = grid_true_heatmaps.max(dim=0, keepdim=True)[0]
        # W, H = 32×32 にリサイズ
        grid_resized = torchvision.transforms.functional.resize(grid, (32, 32))
        true_heatmaps = torchvision.transforms.functional.resize(grid_true_heatmaps, (32, 32))

        # バッチ次元を追加してリストに保存
        output_list.append(grid_resized.unsqueeze(0))  # [1, 1, 32, 32]
        true_heatmaps_list.append(true_heatmaps)  # [1, 1, 32, 32]

    # [N, 1, 32, 32] のテンソルに変換
    return torch.cat(output_list, dim=0), torch.cat(true_heatmaps_list, dim=0)


# 【新規追加】CutMix用のランダムなバウンディングボックスを計算する関数
def rand_bbox(size, lam):
    W = size[3]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W - cut_w // 2)
    cy = np.random.randint(H - cut_h // 2)
    bbx1 = np.clip(cx - cut_w // 2, 0, W - cut_w // 2)
    bby1 = np.clip(cy - cut_h // 2, 0, H - cut_h // 2)
    bbx2 = bbx1 + cut_w
    bby2 = bby1 + cut_h
    # bbx1 = np.clip(cx - cut_w // 2, 0, W)
    # bby1 = np.clip(cy - cut_h // 2, 0, H)
    # bbx2 = np.clip(cx + cut_w // 2, 0, W)
    # bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

# 【新規追加】CutMixの処理
def cutmix_data(x, y, true_heatmaps, alpha=1.0, seed=None, rand_img_bbox=True):
    """
    CutMixの処理を行う関数
    Args:
        x (torch.Tensor): 入力画像 [B, C, H, W]
        y (torch.Tensor): ラベル [B]
        true_heatmaps (torch.Tensor): 真のヒートマップ [B, H, W]
        alpha (float): CutMixのパラメータ
        seed (int): シード値
        rand_img_bbox (bool): バウンディングボックスをランダムに生成するかどうか" or "False" (0 padding)
    Returns:
        x (torch.Tensor): CutMix後の画像 [B, C, H, W]
        y_a (torch.Tensor): ラベルA [B]
        y_b (torch.Tensor): ラベルB [B]
        lam (float): CutMixの係数
        true_heatmaps (torch.Tensor): 更新された真のヒートマップ [B, H, W]
    """
    if seed is not None:
        np.random.seed(seed)  # シードを固定
    if alpha > 0.0:
        lam = alpha
    else:
        lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    if seed is not None:
        rand_index = torch.randperm(batch_size, generator=torch.Generator().manual_seed(seed)).to(x.device)
    else:
        rand_index = torch.randperm(batch_size).to(x.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    # 指定領域をランダムに入れ替え
    if rand_img_bbox:
        x[:, :, bby1:bby2, bbx1:bbx2] = x[rand_index, :, bby1:bby2, bbx1:bbx2]
    else:
        x[:, :, bby1:bby2, bbx1:bbx2] = 0.0
     # **true_heatmaps の更新**
    # updated_heatmaps = true_heatmaps.clone()  # 元の true_heatmaps を変更しないようにコピー
    if (bby2-bby1) * (bbx2-bbx1) >= x.size(2) * x.size(3) // 2:
        true_heatmaps[:, bby1:bby2, bbx1:bbx2] = 1.0
    else:
        # CutMix領域以外を 1 にする（反転する形）
        true_heatmaps = torch.ones_like(true_heatmaps, device=x.device)
        true_heatmaps[:, bby1:bby2, bbx1:bbx2] = 0.0
        # CutMixした領域を 1 にする
    # CutMix によるラベルの混合
    y_a = y
    y_b = y[rand_index]
    return x, y_a, y_b, lam, true_heatmaps

# 検証データの評価（Val Loss と Accuracy の記録を追加）
def evaluate_model(model, dataloader, criterion, writer, epoch):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            if wandb.config["use_cutmix"]:
                true_heatmaps = torch.zeros((images.size(0), images.size(2), images.size(3))).to(device)
                images, labels_a, labels_b, lam, true_heatmaps = cutmix_data(images, labels, true_heatmaps, wandb.config["cutmix_alpha"], seed=wandb.config["seed"])
                # forward pass
                # outputs = model(images)
                # loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                grad_target = labels_a  if lam >= 0.5 else labels_b # gradloss計算には一方のラベルを用いる
            elif wandb.config["use_composite"]:
                images, true_heatmaps = create_composite_image(images)
                labels = labels
                grad_target = labels
            else:
                true_heatmaps = create_mask(images, images.size(2), wandb.config["mask_size"])  # ※通常入力画像サイズに合わせる
                grad_target = labels
            outputs = model(images)
            loss = criterion(outputs, grad_target)

        if batch_idx == 0:
            images.requires_grad = True
            outputs = model(images)  # 再度 forward pass
            loss_metapred = torch.sum(outputs * F.one_hot(grad_target, num_classes=10), dim=-1)
            sa_maps = torch.autograd.grad(loss_metapred, images, grad_outputs=torch.ones_like(loss_metapred), retain_graph=True)[0]
            # sa_maps = gaussian_blur(sa_maps, gaussian_kernel.to(device))  # ガウシアンフィルタ適用
            # if wandb.config["use_cutmix"]:
            sa_maps = torch.mean(sa_maps, dim=1)  # チャンネル方向の平均
            # heatmaps_preprocess = 1 - true_heatmaps
            sa_maps_preprocess = standardize_cut(sa_maps)
            # sa_maps_preprocess = standardize_cut(sa_maps) * heatmaps_preprocess
            # heatmaps_preprocess = standardize_cut(true_heatmaps)
            # heatmaps_preprocess = gaussian_blur(heatmaps_preprocess.unsqueeze(1), gaussian_kernel.to(device)).squeeze(1)  # ガウシアンフィルタ適用
            wandb.log({
                f"Validation Images": [wandb.Image(images[0], caption=f"Epoch {epoch}")],
                f"Validation Saliency Heatmaps": [wandb.Image(sa_maps_preprocess[0:1], caption=f"Epoch {epoch}")],
            }, step=epoch)
            if wandb.config["val_truemap"]:
                wandb.log({
                    f"Validation True Heatmaps": [wandb.Image(sa_maps_preprocess[0:1]*true_heatmaps.unsqueeze(1)[0], caption=f"Epoch {epoch}")]
                }, step=epoch)
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == grad_target).sum().item()
    accuracy = 100 * correct / total
    val_loss /= len(dataloader)
    wandb.log({"val_loss": val_loss,  "epoch": epoch}, step=epoch)
    wandb.log({"val_acc": accuracy,  "epoch": epoch}, step=epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('Accuracy/Validation', accuracy, epoch)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%')

def main(args):
    # wandb の初期化
    wandb.init(project="cnn-cifar10", config=vars(args))

    # 両方を同時に使うと意味が衝突するため
    if wandb.config["use_composite"] and wandb.config["use_cutmix"]:
        raise ValueError("use_compositeとuse_cutmixは同時には使えません。どちらか一方を選択してください。")

    writer = SummaryWriter()

    num_epochs = wandb.config["epochs"]
    use_gradloss = wandb.config["use_gradloss"]
    use_pyramidal_mse = wandb.config["use_pyramidal_mse"]
    channel = wandb.config["channel"]
    mask_size = wandb.config["mask_size"]
    batch_size = wandb.config["batch_size"]
    learning_rate = wandb.config["learning_rate"]
    alpha = wandb.config["alpha"]
    alpha_grad = wandb.config["alpha_grad"]
    alpha_weight = wandb.config["alpha_weight"]

    if channel == 1:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Grayscale (1 channel)
            transforms.Resize((32, 32)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Grayscale (1 channel)　
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif channel == 3:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # RGBの場合は変換なし
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        val_transform = transforms.Compose([
            transforms.Resize((32, 32)),  # RGBの場合は変換なし
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError("Invalid channel setting. Only 1 (Grayscale) or 3 (RGB) is supported.")

    dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataset = CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNNClassifier(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # モデルの学習
    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_loss = 0.0
        epoch_noise_gradient_norm_loss = 0.0
        model.train()
        for batch_idx, (images, labels) in enumerate(dataloader):
            # images, labelsをdeviceへ
            images, labels = images.to(device), labels.to(device)
            # CutMix用の真のヒートマップを作成
            # 以下、configに応じたデータ拡張を適用
            if wandb.config["use_cutmix"]:
                true_heatmaps = torch.zeros((images.size(0), images.size(2), images.size(3))).to(device)
            else:
                true_heatmaps = create_mask(images, images.size(2), mask_size)  # ※通常入力画像サイズに合わせる

            if wandb.config["use_cutmix"]:
                images, labels_a, labels_b, lam, true_heatmaps = cutmix_data(images, labels, true_heatmaps, wandb.config["cutmix_alpha"])
                # forward pass
                images.requires_grad = True
                outputs = model(images)
                grad_target = labels_a  if lam >= 0.5 else labels_b # gradloss計算には一方のラベルを用いる
                # labels = grad_target
                loss = criterion(outputs, labels_a) if lam >= 0.5 else criterion(outputs, labels_b)
                # loss = lam * criterion(outputs, labels_a) if lam >= 0.5 else (1 - lam) * criterion(outputs, labels_b)
                # loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            elif wandb.config["use_composite"]:
                # バッチ内先頭以外を0にマスクし、グリッド画像にまとめる
                images, true_heatmaps = create_composite_image(images)
                # **修正: ラベルを `1/batch_size` の値でロス計算**
                # 1. ラベルを one-hot に変換
                # labels_one_hot = F.one_hot(labels, num_classes=10).float()  # [B, num_classes]
                # 2. すべてのラベルを `1/batch_size` の値に変更
                # labels_smooth = torch.full_like(labels_one_hot, 1 / batch_size)
                images.requires_grad = True
                outputs = model(images)
                # outputs = F.log_softmax(outputs, dim=-1)  # KLDivLoss のために log_softmax を適用
                # criterion = nn.KLDivLoss(reduction="batchmean")  # `reduction="batchmean"` で正規化
                # loss = criterion(outputs, labels_smooth)
                loss = criterion(outputs, labels)
                # loss = criterion(outputs, labels) / batch_size
                grad_target = labels
            else:
                images.requires_grad = True
                outputs = model(images)
                loss = criterion(outputs, labels)
                grad_target = labels

            # 出力とラベルに対してone-hot重み付き和をとり、画像に対する勾配を計算
            loss_metapred = torch.sum(outputs * F.one_hot(grad_target, num_classes=10), dim=-1)
            sa_maps = torch.autograd.grad(loss_metapred, images, grad_outputs=torch.ones_like(loss_metapred), retain_graph=True)[0]
            # sa_maps = gaussian_blur(sa_maps, gaussian_kernel.to(device))  # ガウシアンフィルタ適用
            # 標準化・カット操作（harmonization loss用）
            if wandb.config["channel"] > 1:
                sa_maps = torch.mean(sa_maps, dim=1)  # チャンネル方向の平均
            heatmaps_preprocess = 1 - true_heatmaps
            sa_maps_preprocess = standardize_cut(sa_maps)
            noise_gradient_norm = sa_maps_preprocess * heatmaps_preprocess
            # heatmaps_preprocess = standardize_cut(true_heatmaps)
            # gaussian_kernel = gaussian_kernel().to(device)
            # heatmaps_preprocess = gaussian_blur(heatmaps_preprocess.unsqueeze(1), gaussian_kernel.to(device)).squeeze(1)  # ガウシアンフィルタ適用
            # _sa_max = torch.amax(sa_maps_preprocess.detach(), dim=(1, 2), keepdim=True)[0] + 1e-6
            # _hm_max = torch.amax(heatmaps_preprocess, dim=(1, 2), keepdim=True)[0] + 1e-6
            # heatmaps_preprocess = heatmaps_preprocess / _hm_max * _sa_max
            if use_gradloss:
                if use_pyramidal_mse:
                    tokens = torch.ones(len(images)).to(device)
                    noise_gradient_norm = pyramidal_mse_with_tokens(torch.zeros_like(heatmaps_preprocess), noise_gradient_norm, tokens)
                else:
                    # max_norm = noise_gradient_norm.max()
                    # noise_gradient_norm /= max_norm  # 最大値で正規化
                    # noise_gradient_norm = torch.norm(noise_gradient_norm, p=2)
                    noise_gradient_norm = torch.nn.functional.mse_loss(noise_gradient_norm, torch.zeros_like(noise_gradient_norm))

                # else:
                #     harmonization_loss = F.mse_loss(sa_maps_preprocess, heatmaps_preprocess)
                # total_loss = loss + alpha * harmonization_loss
                # noise_gradient_norm = noise_gradient_norm / loss.max()[0] * noise_gradient_norm.max()[0]
                weight_loss = sum(torch.norm(param, p=2) for name, param in model.named_parameters()
                        if not any(exclude in name for exclude in ['bn', 'normalization', 'embed', 'Norm', 'norm', 'class_token']))
                total_loss = alpha * loss + alpha_grad * noise_gradient_norm + alpha_weight * weight_loss
                if wandb.config["loss_norm"]:
                    total_loss = loss_norm(total_loss)

                # total_loss = loss + alpha * harmonization_loss + lambda_weights * weight_loss
            else:
                total_loss = loss
            # # 5エポックごとに可視化
            if (epoch % 5 == 0 or epoch == num_epochs-1) and batch_idx == 0:
                # WandB に追加
                wandb.log({
                    f"Images": [wandb.Image(images[0], caption=f"Epoch {epoch}")],
                    f"Saliency Heatmaps": [wandb.Image(sa_maps_preprocess[0:1], caption=f"Epoch {epoch}")],
                    f"True Heatmaps": [wandb.Image(sa_maps_preprocess[0:1]*true_heatmaps.unsqueeze(1)[0], caption=f"Epoch {epoch}")]
                }, step=epoch)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_total_loss += total_loss.item()
            if use_gradloss:
                noise_gradient_norm = noise_gradient_norm.detach() * alpha
                epoch_noise_gradient_norm_loss += noise_gradient_norm.item()

        average_loss = epoch_total_loss / len(dataloader)
        average_classify_loss = epoch_loss / len(dataloader)
        average__noise_gradient_norm_loss = epoch_noise_gradient_norm_loss / len(dataloader)
        scheduler.step(average_loss)
        wandb.log({"train_loss": average_loss, "lr": optimizer.param_groups[0]['lr'], "epoch": epoch}, step=epoch)
        if use_gradloss:
            wandb.log({"classify_loss": average_classify_loss, "lr": optimizer.param_groups[0]['lr'], "epoch": epoch}, step=epoch)
            wandb.log({"noise_grad_loss": average__noise_gradient_norm_loss, "lr": optimizer.param_groups[0]['lr'], "epoch": epoch}, step=epoch)
        writer.add_scalar("Loss/train", average_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        if (epoch % 5 == 0 or epoch == num_epochs-1):
            if epoch == num_epochs-1:
                    print(wandb.config["seed"])
            evaluate_model(model, val_dataloader, criterion, writer, epoch)
    writer.close()
    wandb.finish()

# コマンドライン引数の設定
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10 with configurable options.")

    # 基本ハイパーパラメータ
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--channel", type=int, choices=[1, 3], default=3, help="Number of channels (1 for grayscale, 3 for RGB).")

    # Augmentation オプション
    parser.add_argument("--use_gradloss", action="store_true", help="Use gradient-based loss.")
    parser.add_argument("--use_pyramidal_mse", action="store_true", help="Use pyramidal MSE loss.")
    parser.add_argument("--use_composite", action="store_true", help="Apply composite image masking augmentation.")
    parser.add_argument("--use_cutmix", action="store_true", help="Use CutMix augmentation.")
    parser.add_argument("--loss_norm", action="store_true", help="Loss normalization.")
    parser.add_argument("--val_truemap", action="store_true", help="Loss normalization.")

    # その他のパラメータ
    parser.add_argument("--mask_size", type=int, default=16, help="Mask size for augmentation.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight for Cross Entorypy loss.")
    parser.add_argument("--alpha_grad", type=float, default=1.0, help="Weight for gradient loss.")
    parser.add_argument("--alpha_weight", type=float, default=0.0, help="Weight for harmonization loss.")
    parser.add_argument("--cutmix_alpha", type=float, default=-1.0, help="CutMix interpolation parameter.")

    args = parser.parse_args()
    main(args)

