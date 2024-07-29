import pandas as pd
import os

def filter_predictions_to_dataframe(file_path):
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
    df_label0_pred1, df_label1_pred0 = filter_predictions_to_dataframe(os.path.join('MobileNetV2_final','MobileNetV2_final_predictions.txt'))

    df_label0_pred1.to_csv(os.path.join('MobileNetV2_final','label0_prediction1.csv'), index=False)
    df_label1_pred0.to_csv(os.path.join('MobileNetV2_final','label1_prediction0.csv'), index=False)
    return df_label0_pred1, df_label1_pred0




