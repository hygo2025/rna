# trainer.py
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import os


class Trainer:
    def __init__(self, model, dataloaders, loss_fn, optimizer, config, experiment_name, writer: SummaryWriter):
        self.model = model
        self.train_loader = dataloaders['train']
        self.val_loader = dataloaders['val']
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.config = config

        self.history = []
        self.best_val_acc = 0.0
        self.early_stop_counter = 0
        self.model_save_path = os.path.join(config.MODEL_SAVE_DIR, f"{experiment_name}.pth")
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

        self.writer = writer

    def _train_step(self):
        self.model.train()

        running_loss = 0.0
        all_preds = []
        all_labels = []

        for data in tqdm(self.train_loader):
            inputs, labels = data
            all_labels.extend(labels)

            inputs = inputs.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            all_preds.extend(list(np.argmax(outputs.cpu().detach().numpy(), axis=-1)))

            running_loss += loss.cpu().item()

        return all_labels, all_preds, running_loss

    def _eval_step(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in tqdm(self.val_loader):
                images, labels = data
                all_labels.extend(labels)

                # calculate outputs by running images through the network
                images = images.to(self.config.DEVICE)
                outputs = self.model(images).cpu().numpy()
                all_preds.extend(list(np.argmax(outputs, axis=-1)))

        return all_labels, all_preds

    def train(self):
        for epoch in range(self.config.NUM_EPOCHS):
            true, pred, loss = self._train_step()
            train_acc = accuracy_score(true, pred)

            true, pred = self._eval_step()
            test_acc = accuracy_score(true, pred)


            self.history.append([loss, train_acc, test_acc])
            print(f"Epoch {epoch} Loss: {loss:.2f} Train acc: {train_acc:.2f} Test acc: {test_acc:.2f} ")

            if test_acc > self.best_val_acc:
                print(f"Acurácia de validação melhorou ({self.best_val_acc:.4f} --> {test_acc:.4f}). Salvando modelo...")
                self.best_val_acc = test_acc
                torch.save(self.model.state_dict(), self.model_save_path)
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            if self.early_stop_counter >= self.config.PATIENCE:
                print(f"\nEarly stopping na época {epoch + 1}.");
                break

            self.writer.add_scalar("loss", loss, epoch)
            self.writer.add_scalars(main_tag="accuracy", tag_scalar_dict={"train": train_acc, "test": test_acc}, global_step=epoch)
        self.writer.close()

        print(f"\nTreinamento concluído. Melhor acurácia de validação: {self.best_val_acc:.4f}")
        return self.history