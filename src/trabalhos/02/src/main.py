# main.py
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import time
import argparse
import os

from config import Config
from data_handler import DataHandler
from models import SimpleCNN, get_pretrained_model
from trainer import Trainer
from reporting import ReportGenerator
from preprocess import Preprocessor


def run_experiment(config: Config, model_type: str, feature_extract: bool):
    timestamp = int(time.time())
    experiment_name = f"{model_type}_{'feature_extract' if feature_extract else 'fine_tune'}_{timestamp}"
    print(f"--- Iniciando Experimento: {experiment_name} ---")

    # 1. Dados
    data_handler = DataHandler(config)
    dataloaders = {
        "train": data_handler.train_loader,
        "val": data_handler.val_loader,
        "test": data_handler.test_loader
    }
    num_classes = len(data_handler.class_names)

    # 2. Modelo
    if model_type == 'custom':
        model = SimpleCNN(num_classes=num_classes)
    else:
        model = get_pretrained_model(model_name=model_type, num_classes=num_classes, feature_extract=feature_extract)
    model.to(config.DEVICE)
    model = torch.compile(model)

    class_weights = config.class_weights_train()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    writer = SummaryWriter(log_dir=os.path.join(config.TENSORBOARD_DIR, experiment_name))

    # 4. Trainer
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        loss_fn=loss_fn,
        optimizer=optimizer,
        config=config,
        experiment_name=experiment_name,
        writer=writer
    )
    history = trainer.train()

    # 5. Relatórios
    print("\nCarregando melhor modelo para avaliação e relatórios finais...")
    model.load_state_dict(torch.load(trainer.model_save_path))

    report_dir = os.path.join(config.REPORTS_DIR, experiment_name)
    reporter = ReportGenerator(model, config.DEVICE, data_handler.class_names, report_dir)
    reporter.plot_curves(history)
    reporter.generate_final_report(dataloaders['test'], loss_fn)
    print(f"--- Experimento {experiment_name} concluído ---")


if __name__ == '__main__':
    # setup
    base_dataset_path = f"/home/{os.environ['USER']}/Documents/datasets"
    config_instance = Config(download_path=base_dataset_path)
    Preprocessor(config=config_instance).run()
    # -----------

    parser = argparse.ArgumentParser(description="Treinamento de Modelos de Raio-X")
    parser.add_argument("--model", type=str, default="custom", choices=["custom", "resnet34"],
                        help="Tipo de modelo a ser treinado.")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "feature_extract"],
                        help="Modo de treinamento para modelos pré-treinados.")
    args = parser.parse_args()

    feature_extract_flag = True if args.mode == "feature_extract" else False

    run_experiment(config=config_instance, model_type=args.model, feature_extract=feature_extract_flag)

    # Exemplo de como rodar os outros modelos via linha de comando:
    # python main.py --model resnet34 --mode feature_extract
    # python main.py --model resnet34 --mode full