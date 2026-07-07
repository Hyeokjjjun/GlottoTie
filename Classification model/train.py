import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from tqdm import tqdm
import numpy as np

# Use the Agg backend so figures can be saved in headless (server) environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import config
from dataloader import FewShotDataset, StrongAugmentation, set_seed
from model import Encoder, PrototypicalNetwork


def save_curves(history, save_path):
    """Save train/val loss and accuracy curve values (JSON) + figures (PNG/PDF)."""
    os.makedirs(save_path, exist_ok=True)

    # 1) Save values (so the curves can be re-plotted later)
    with open(os.path.join(save_path, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # 2) Save figures
    ep = history["episode"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(ep, history["train_loss"], label="train", marker="o", ms=3)
    axes[0].plot(ep, history["val_loss"], label="validation", marker="s", ms=3)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["train_acc"], label="train", marker="o", ms=3)
    axes[1].plot(ep, history["val_acc"], label="validation", marker="s", ms=3)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training / Validation Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_curves.png"), dpi=300,
                bbox_inches="tight")
    plt.savefig(os.path.join(save_path, "training_curves.pdf"), bbox_inches="tight")
    plt.close()


def train():
    # Set random seed
    set_seed(config.SEED)

    # Data augmentation
    if config.USE_AUGMENTATION:
        augmentation = StrongAugmentation(prob=config.AUGMENTATION_PROB)
    else:
        augmentation = None

    # Datasets (3-class)
    train_dataset = FewShotDataset(
        config.COMMERCIAL_DATA_PATH,
        augmentation=augmentation,
        num_classes=config.NUM_CLASSES
    )
    val_dataset = FewShotDataset(
        config.OUR_DEVICE_DATA_PATH,
        augmentation=None,
        num_classes=config.NUM_CLASSES
    )

    # Model
    encoder = Encoder(
        input_channels=config.INPUT_CHANNELS,
        embedding_dim=config.EMBEDDING_DIM,
        dropout=config.DROPOUT_RATE
    ).to(config.DEVICE)

    model = PrototypicalNetwork(encoder).to(config.DEVICE)

    # Number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Checkpoint directory
    os.makedirs(config.SAVE_PATH, exist_ok=True)

    # Training
    print("\n" + "="*60)
    print(f"Training Start: 3-way {str(config.K_SHOT)}-shot Learning")
    print("="*60 + "\n")

    best_val_acc = 0.0
    train_losses = []
    train_accs = []

    # For recording curves (train/val loss and acc)
    history = {"episode": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    for episode in tqdm(range(config.NUM_EPISODES), desc="Training"):
        model.train()

        # Episode sampling
        support, support_labels, query, query_labels = train_dataset.sample_episode(
            n_way=config.N_WAY,
            k_shot=config.K_SHOT,
            n_query=config.N_QUERY
        )

        support = support.to(config.DEVICE)
        support_labels = support_labels.to(config.DEVICE)
        query = query.to(config.DEVICE)
        query_labels = query_labels.to(config.DEVICE)

        # Forward
        logits = model(support, support_labels, query, config.N_WAY, config.K_SHOT)
        loss = criterion(logits, query_labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Accuracy
        _, predicted = torch.max(logits, 1)
        accuracy = (predicted == query_labels).float().mean()

        train_losses.append(loss.item())
        train_accs.append(accuracy.item())

        # Periodic logging
        if (episode + 1) % config.PRINT_EVERY == 0:
            avg_loss = np.mean(train_losses[-config.PRINT_EVERY:])
            avg_acc = np.mean(train_accs[-config.PRINT_EVERY:])
            current_lr = optimizer.param_groups[0]['lr']

            print(f"\nEpisode [{episode+1}/{config.NUM_EPISODES}]")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Train Acc:  {avg_acc:.4f} ({avg_acc*100:.2f}%)")
            print(f"  LR:         {current_lr:.6f}")

        # Periodic validation (compute both loss and acc) + record curves
        if (episode + 1) % config.VAL_EVERY == 0:
            val_loss, val_acc = validate(model, val_dataset, criterion)

            # Also record the mean train loss and acc over the same interval
            train_loss_avg = float(np.mean(train_losses[-config.VAL_EVERY:]))
            train_acc_avg = float(np.mean(train_accs[-config.VAL_EVERY:]))

            history["episode"].append(episode + 1)
            history["train_loss"].append(train_loss_avg)
            history["train_acc"].append(train_acc_avg)
            history["val_loss"].append(float(val_loss))
            history["val_acc"].append(float(val_acc))

            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_acc:.4f} ({val_acc*100:.2f}%)")

            # Save the model with the highest validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'episode': episode + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, os.path.join(config.SAVE_PATH, 'best_model.pth'))
                print(f"  ✓ Best model saved! (Val Acc: {val_acc*100:.2f}%)")

    # Save the final model
    torch.save({
        'episode': episode + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(config.SAVE_PATH, 'final_model.pth'))

    # Save curves (values as JSON + figures as PNG/PDF)
    save_curves(history, config.SAVE_PATH)
    print(f"\n[saved] training curves -> {config.SAVE_PATH}/training_curves.png/.pdf")
    print(f"[saved] training history -> {config.SAVE_PATH}/training_history.json")

    print("\n" + "="*60)
    print(f"Training Completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Final Episode: {episode + 1}")
    print("="*60)


def validate(model, dataset, criterion):
    """Validation (returns both loss and acc)"""
    model.eval()
    losses = []
    accuracies = []

    with torch.no_grad():
        for _ in range(config.VAL_EPISODES):
            support, support_labels, query, query_labels = dataset.sample_episode(
                n_way=config.N_WAY,
                k_shot=config.K_SHOT,
                n_query=config.N_QUERY
            )

            support = support.to(config.DEVICE)
            support_labels = support_labels.to(config.DEVICE)
            query = query.to(config.DEVICE)
            query_labels = query_labels.to(config.DEVICE)

            logits = model(support, support_labels, query, config.N_WAY, config.K_SHOT)
            loss = criterion(logits, query_labels)

            _, predicted = torch.max(logits, 1)
            accuracy = (predicted == query_labels).float().mean()

            losses.append(loss.item())
            accuracies.append(accuracy.item())

    return np.mean(losses), np.mean(accuracies)


if __name__ == '__main__':
    train()