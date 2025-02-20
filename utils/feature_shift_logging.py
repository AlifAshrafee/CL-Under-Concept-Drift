import torch


def extract_features(model, dataloader, file_path):

    status = model.net.training
    model.net.eval()

    features_list = []
    labels_list = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0], data[1]
            inputs = inputs.to(model.device)
            features = model.net(inputs, returnt="features")
            features_list.append(features.cpu())
            labels_list.append(labels)

    features = torch.cat(features_list)
    labels = torch.cat(labels_list)

    print("Extracted Features Shape: ", features.shape)
    print("Labels Shape: ", labels.shape)

    torch.save({"features": features, "labels": labels}, file_path)
    print("Features saved successfully")

    model.net.train(status)
    return
