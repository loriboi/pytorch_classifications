import torch
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
def trainprocess(hyperp, dataset, early_stopping_patience):
    report = []
    hyperp.saveModelInfo()
    early_stopping_counter = 0
    best_val_loss = float('inf')
    for epoch in range(hyperp.num_epochs):
        print(f"Epoch {epoch}/{hyperp.num_epochs-1}")
        hyperp.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in dataset.train_loader:  # Ignora il nome dell'immagine qui
            inputs, labels = inputs.to(hyperp.device), labels.to(hyperp.device)  # Trasferisci inputs e labels sul dispositivo
            hyperp.optimizer.zero_grad()
            outputs = hyperp.model(inputs)
            loss = hyperp.criterion(outputs, labels)
            loss.backward()
            hyperp.optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(dataset.train_loader.dataset)
        train_accuracy = 100. * train_correct / train_total
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Valutazione sul set di validazione
        hyperp.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in dataset.val_loader:  # Ignora il nome dell'immagine qui
                inputs, labels = inputs.to(hyperp.device), labels.to(hyperp.device)  # Trasferisci inputs e labels sul dispositivo
                outputs = hyperp.model(inputs)
                loss = hyperp.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(dataset.val_loader.dataset)
        val_accuracy = 100. * val_correct / val_total
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        report.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        if hyperp.schedulerON:
            hyperp.scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': hyperp.model.state_dict(),
                'optimizer_state_dict': hyperp.optimizer.state_dict(),
                'scheduler_state_dict': hyperp.scheduler.state_dict() if hyperp.schedulerON else None,
                'loss': best_val_loss,
            }, hyperp.checkpoint_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    with open(hyperp.latimeline, 'w') as file:
        json.dump(report, file, indent=4)
    hyperp.shutdownatend(False)

def testmodel(hyperp, dataset):
    all_labels = []
    all_preds = []
    all_image_ids = []  # Lista per salvare gli ID delle immagini

    hyperp.model.to(hyperp.device)
    hyperp.model.eval()

    with torch.no_grad():
        for data, labels in dataset.test_loader:
            data, labels = data.to(hyperp.device), labels.to(hyperp.device)
            outputs = hyperp.model(data)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Aggiungi gli ID delle immagini alla lista
            all_image_ids.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds)
    print(report)
    report_path = os.path.join(hyperp.name, hyperp.name + "_report.txt")
    save_classification_report(report, report_path)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    confusion_matrix_path = os.path.join(hyperp.name, hyperp.name + "_confusionmatrix.png")
    plt.savefig(confusion_matrix_path)
    plt.show()

    predictions_path = os.path.join(hyperp.name, hyperp.name + "_predictions.txt")
    save_predictions(all_labels, all_preds, predictions_path)

    # Salva gli ID delle immagini in un file separato
    image_ids_path = os.path.join(hyperp.name, hyperp.name + "_image_ids.txt")
    save_image_ids(all_image_ids, image_ids_path)
    print("Test DONE")

def save_image_ids(image_ids, filename):
    with open(filename, 'w') as f:
        for i, img_id in enumerate(image_ids):
            f.write(f"Index {i}: Image ID: {img_id}\n")


def save_predictions(all_labels, all_preds, filename):
    with open(filename, 'w') as f:
        for i, (label, pred) in enumerate(zip(all_labels, all_preds)):
            if label == pred:
                f.write(f"Index {i}: Correct - Label: {label}, Prediction: {pred}\n")
            else:
                f.write(f"Index {i}: Incorrect - Label: {label}, Prediction: {pred}\n")

def draw_curves(hyperp):
    # Leggere i dati dal file txt
    with open(hyperp.latimeline, 'r') as f:
        data = json.load(f)

    # Convertire i dati in un DataFrame pandas
    df = pd.DataFrame(data)

    # Plot delle perdite
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(hyperp.name,hyperp.name+"_losscurves.png"))
    # Plot delle accuratezze
    plt.figure(figsize=(10, 5))
    plt.plot(df['epoch'], df['train_accuracy'], label='Train Accuracy')
    plt.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(hyperp.name,hyperp.name+"_acccurves.png"))
    
    print("Curves SAVED")

def save_classification_report(report, filename):
    with open(filename, 'w') as f:
        f.write(report)


def filter_predictions_to_dataframe(root_path,file_path):
    # Liste per conservare i dati filtrati
    label0_pred1 = []
    label1_pred0 = []

    # Apertura del file di input
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Iterazione su ogni riga del file
    for line in lines:
        try:
            # Estrazione di indice, label e prediction
            index_str = line.split('Index ')[1].split(':')[0].strip()
            index = int(index_str)

            label_str = line.split('Label: ')[1].split(',')[0].strip()
            label = int(label_str)

            prediction_str = line.split('Prediction: ')[1].strip()
            prediction = int(prediction_str)
        except (IndexError, ValueError):
            print(f"Skipping line due to parsing error: {line}")
            continue

        # Verifica delle condizioni e aggiunta alle liste appropriate
        if label == 0 and prediction == 1:
            label0_pred1.append((index, label, prediction))
        elif label == 1 and prediction == 0:
            label1_pred0.append((index, label, prediction))

    # Creazione dei DataFrame
    df_label0_pred1 = pd.DataFrame(label0_pred1, columns=['Index', 'Label', 'Prediction'])
    df_label1_pred0 = pd.DataFrame(label1_pred0, columns=['Index', 'Label', 'Prediction'])

    df_label0_pred1.to_csv(os.path.join(root_path,'label0_prediction1.csv'), index=False)
    df_label1_pred0.to_csv(os.path.join(root_path,'label1_prediction0.csv'), index=False)
    return df_label0_pred1, df_label1_pred0

def plot_images_test(test_set,dflab0pred1,dflab1pred0):
    return