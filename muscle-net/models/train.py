import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier

from config import (
    DEVICE,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    TRAIN_SPLIT,
    SEEDS,
    S1_DIR,
    S2_DIR,
    LABEL_DIR,
    NUM_CLASSES,
    MAIN_LOSS_WEIGHT,
    AUX_LOSS_WEIGHT,
)
from data.dataset import Sentinel2Dataset
from models.muscle_net import MUSCLENet
from utils.losses import DeepSupervisionLoss
from utils.metrics import calculate_iou
from utils.seed import set_seed, seed_worker_factory


def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=200, save_path="best_model.pth", patience=20):
    model.to(device)
    best_val_iou = 0.0
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_iou_main = 0.0
        train_iou_aux = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            main_output, aux_output = model(images)
            loss = criterion(main_output, aux_output, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_iou_main += calculate_iou(main_output, labels).item()
            train_iou_aux += calculate_iou(aux_output, labels).item()

        model.eval()
        val_loss = 0.0
        val_iou_main = 0.0
        val_iou_aux = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                main_output, aux_output = model(images)
                loss = criterion(main_output, aux_output, labels)

                val_loss += loss.item()
                val_iou_main += calculate_iou(main_output, labels).item()
                val_iou_aux += calculate_iou(aux_output, labels).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        avg_train_iou_main = train_iou_main / len(train_loader)
        avg_train_iou_aux = train_iou_aux / len(train_loader)

        avg_val_iou_main = val_iou_main / len(val_loader)
        avg_val_iou_aux = val_iou_aux / len(val_loader)

        final_val_iou = (MAIN_LOSS_WEIGHT * avg_val_iou_main) + (AUX_LOSS_WEIGHT * avg_val_iou_aux)

        if final_val_iou > best_val_iou:
            best_val_iou = final_val_iou
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at epoch {epoch + 1} with IoU: {best_val_iou:.4f}")
        else:
            bad_epochs += 1
            print(f"No improvement at epoch {epoch + 1} (bad epochs: {bad_epochs})")

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Train IoU - Main: {avg_train_iou_main:.4f}, Aux: {avg_train_iou_aux:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val IoU - Main: {avg_val_iou_main:.4f}, Aux: {avg_val_iou_aux:.4f}")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Best model was at epoch {best_epoch + 1} with IoU: {best_val_iou:.4f}")


def main():
    for seed in SEEDS:
        print(f"\n== Training with seed {seed} ==")

        generator = set_seed(seed)
        seed_worker = seed_worker_factory(seed)

        dataset = Sentinel2Dataset(S1_DIR, S2_DIR, LABEL_DIR)

        train_size = int(TRAIN_SPLIT * len(dataset))
        val_size = len(dataset) - train_size

        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            generator=generator,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            generator=generator,
        )

        pretrained_model = BigEarthNetv2_0_ImageClassifier.from_pretrained(
            "BIFOLD-BigEarthNetv2-0/resnet50-all-v0.2.0"
        )

        model = MUSCLENet(pretrained_model, num_classes=NUM_CLASSES)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = DeepSupervisionLoss(
            main_weight=MAIN_LOSS_WEIGHT,
            aux_weight=AUX_LOSS_WEIGHT,
        )

        save_path = f"DFC2020_model_seed{seed}.pth"

        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=DEVICE,
            num_epochs=NUM_EPOCHS,
            save_path=save_path,
            patience=PATIENCE,
        )


if __name__ == "__main__":
    main()
