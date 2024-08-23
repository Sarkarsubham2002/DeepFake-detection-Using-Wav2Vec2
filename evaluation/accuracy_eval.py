import librosa
from datasets import load_dataset
from transformers import AutoConfig, Wav2Vec2Processor
from sklearn.metrics import classification_report
import torch



test_dataset = load_dataset("csv", data_files={"test": "/kaggle/input/iwa-testtrain/test.csv"}, delimiter="\t")["test"]
# print(test_dataset)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device: {device}")

# # config
# config = AutoConfig.from_pretrained(
#     model_name_or_path,
#     num_labels=num_labels,
#     label2id={label: i for i, label in enumerate(label_list)},
#     id2label={i: label for i, label in enumerate(label_list)},
#     finetuning_task="wav2vec2_clf",
# )
# setattr(config, 'pooling_mode', "mean")

# model_name_or_path = "/kaggle/working/dfmodel"
model_name_or_path = "path to the trained model in the hugging face(where the model is stored)"
config = AutoConfig.from_pretrained(model_name_or_path)
processor = Wav2Vec2Processor.from_pretrained(model_name_or_path1)
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path,config=config).to(device)



def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
#     speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, processor.feature_extractor.sampling_rate)
    speech_array = librosa.resample(y=np.asarray(speech_array), orig_sr=sampling_rate, target_sr=processor.feature_extractor.sampling_rate)

    batch["speech"] = speech_array
    return batch


def predict(batch):
    # Process the batch
    features = processor(batch["speech"], sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)

    # Check if attention_mask exists before trying to use it
    if "attention_mask" in features:
        attention_mask = features.attention_mask.to(device)
    else:
        attention_mask = None

    # Perform inference
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits 

    # Get the predicted class
    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()

    # Store the logits and predictions in the batch
    batch["predicted"] = pred_ids
    batch["logits"] = logits.detach().cpu().numpy()

    return batch



test_dataset = test_dataset.map(speech_file_to_array_fn)

# Assuming you're using this in a dataset.map or similar:
predicted_results = test_dataset.map(predict, batched=True, batch_size=8)

# You can access logits and binary predictions from the predicted_results:
logits = predicted_results["logits"]
predictions = predicted_results["predicted"]


label_names = [config.id2label[i] for i in range(config.num_labels)]
# label_names


y_true = [config.label2id[name] for name in  test_dataset["label"]]
y_pred = predictions

# print(y_true[:5])
# print(y_pred[:5])


# print(classification_report(y_true, y_pred, target_names=label_names))

