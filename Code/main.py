import os
import json
import pickle
import torch
import utils.data_processing as data_processing
import model.model as model

def main():
    print("--- INITIALIZATION ---")
    data_path = "..\\Data\\turkish_data\\MAWPS-ASDIV-SWAMP".replace("\\", os.sep)
    vocabulary_path = "..\\Data\\vocabulary\\MAWPS-ASDIV-SWAMP".replace("\\", os.sep)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n\tDevice: ", device)

    # split_dataset(data_path, train_size=3301)
    # create_vocabularies(data_path, vocabulary_path)

    word_voc, equation_voc = read_vocabularies(vocabulary_path)
    seq2seq_model = model.Seq2Seq(word_voc, equation_voc, device)
    seq2seq_model = seq2seq_model.to(device)

    print("--- INITIALIZATION DONE ---\n\n--- TRAINING ---")
    model.train(seq2seq_model, word_voc, equation_voc, read_data_loader("train", vocabulary_path), read_data_loader("dev", vocabulary_path), device)
    print("\n\n--- TRAINING DONE ---")

def split_dataset(data_path, train_size):
    splitted_dev_train_path = "..\\Data\\turkish_data\\MAWPS-ASDIV-SWAMP"
    with open(os.path.join(data_path, "data.json"), 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    data_processing.random_and_split_dataset(data, splitted_dev_train_path, train_size)

def create_vocabularies(data_path, vocabulary_path):
    train_data_loader, dev_data_loader = create_dataloaders(data_path, vocabulary_path)

    print("\tCreating dictionary for questions...")
    word_voc = data_processing.WordVoc()
    word_voc.create_word_vocabulary(dataloaders=[train_data_loader, dev_data_loader])
    data_processing.save_dict_as_pkl(output_path=os.path.join(vocabulary_path, "question_vocabulary.pkl"), vocabulary=word_voc)
    print("\tQuestion dictionary is saved as a pickle file.")

    print("\tCreating dictionary for equations...")
    equation_voc = data_processing.EquationVoc()
    equation_voc.create_equation_vocabulary(dataloaders=[train_data_loader, dev_data_loader])
    data_processing.save_dict_as_pkl(output_path=os.path.join(vocabulary_path, "equation_vocabulary.pkl"), vocabulary=equation_voc)
    print("\tEquation dictionary is saved as a pickle file.\n")

def create_dataloaders(data_path, vocabulary_path):
    print("\tLoading train and dev data...")

    train_data_loader = data_processing.read_data(path=os.path.join(data_path, "translated_train_data.json"))
    data_processing.save_dict_as_pkl(output_path=os.path.join(vocabulary_path, "train_data_loader.pkl"), vocabulary=train_data_loader)

    dev_data_loader = data_processing.read_data(path=os.path.join(data_path, "translated_dev_data.json"))
    data_processing.save_dict_as_pkl(output_path=os.path.join(vocabulary_path, "dev_data_loader.pkl"), vocabulary=dev_data_loader)

    return train_data_loader, dev_data_loader

def read_vocabularies(vocabulary_path):
    with open(os.path.join(vocabulary_path, "question_vocabulary.pkl"), 'rb') as handle:
        wordVoc = pickle.load(handle)
    with open(os.path.join(vocabulary_path, "equation_vocabulary.pkl"), 'rb') as handle:
        equationVoc = pickle.load(handle)
    return wordVoc, equationVoc

def read_data_loader(data, vocabulary_path):
    with open(os.path.join(vocabulary_path, "{}_data_loader.pkl".format(data)), 'rb') as handle:
        data = pickle.load(handle)
    return data

if __name__ == "__main__":
    main()