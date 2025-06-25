from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


def train(model, loader, optimizer, criterion, device, is_NCA=False):
    model.train()
    total_loss, total_correct = 0, 0
    loop = tqdm(loader, desc="Training", leave=False)

    for x, y in loop:
        x, y = x.to(device), y.squeeze().to(device)
        optimizer.zero_grad()
        out = model(x) if not is_NCA else model(x)[0]
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / len(loader.dataset)
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, is_NCA=False):
    model.eval()
    total_loss, total_correct = 0, 0
    loop = tqdm(loader, desc="Evaluating", leave=False)

    for x, y in loop:
        x, y = x.to(device), y.squeeze().to(device)
        out = model(x) if not is_NCA else model(x)[0]
        loss = criterion(out, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (out.argmax(1) == y).sum().item()

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader.dataset)
    avg_acc = total_correct / len(loader.dataset)
    return avg_loss, avg_acc


def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies, save_path):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Acc')
    plt.plot(epochs, val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
