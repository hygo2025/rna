# reporting.py
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


class ReportGenerator:
    def __init__(self, model, device, class_names, report_dir):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

    def plot_curves(self, history):
        history = np.array(history)
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1);
        plt.title('Acurácia ao Longo das Épocas')
        plt.plot(history[:, 1], '-o', label='Treino');
        plt.plot(history[:, 3], '-o', label='Validação')
        plt.xlabel('Época');
        plt.ylabel('Acurácia');
        plt.legend();
        plt.grid(True)
        plt.subplot(1, 2, 2);
        plt.title('Perda (Loss) ao Longo das Épocas')
        plt.plot(history[:, 0], '-o', label='Treino');
        plt.plot(history[:, 2], '-o', label='Validação')
        plt.xlabel('Época');
        plt.ylabel('Perda');
        plt.legend();
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.report_dir, "learning_curves.png"))
        plt.show()

    def generate_final_report(self, test_loader, loss_fn):
        self.model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Gerando relatório final..."):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy());
                all_labels.extend(labels.cpu().numpy())

        print(f"\n--- Relatório de Classificação Final ---")
        report = classification_report(all_labels, all_preds, target_names=self.class_names)
        print(report)
        with open(os.path.join(self.report_dir, "classification_report.txt"), 'w') as f:
            f.write(report)

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(self.class_names))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap='cividis', xticks_rotation='vertical')
        plt.grid(False);
        plt.title('Matriz de Confusão Final')
        plt.savefig(os.path.join(self.report_dir, "confusion_matrix.png"))
        plt.show()