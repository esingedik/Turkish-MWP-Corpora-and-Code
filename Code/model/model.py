import os
import sys
from os import path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import utils.param as param
import model.encoder as encoder
import model.decoder as decoder
import model.embedding as embedding
import utils.data_processing as data_processing

param = param.param()
actual_predicted_equations_voc = {"+ number0 number1": 0}

class Seq2Seq(nn.Module):
    def __init__(self, wordVoc, equationVoc, device):
        super(Seq2Seq, self).__init__()
        print("\tInitializing Model...")

        self.param = param
        self.epoch = self.param.epoch
        self.wordVoc = wordVoc
        self.equationVoc = equationVoc
        self.equationEmbedding = nn.Embedding(num_embeddings=self.equationVoc.number_of_distinct_operators, embedding_dim=self.param.embedding_dimension_for_equations)

        nn.init.uniform_(tensor=self.equationEmbedding.weight, a=-self.param.uniform_dist_bound, b=self.param.uniform_dist_bound)

        self.device = device
        self.wordEmbedding = embedding.Embedding(self.device, self.wordVoc)
        self.encoder = encoder.Encoder(device)
        self.decoder = decoder.Attention_Decoder(embedding=self.equationEmbedding, out_features=self.equationVoc.number_of_distinct_operators).to(device)

        self.NLLLoss = nn.NLLLoss()
        self.adam_optimizer = optim.Adam(
            [{"params": self.wordEmbedding.parameters(), "lr": self.param.embedding_learning_rate},
             {"params": self.encoder.parameters()},
             {"params": self.decoder.parameters()}], lr=self.param.learning_rate)

        self.parameters = list(self.wordEmbedding.parameters()) + list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.validation_loss = torch.tensor(float('inf')).item()

        self.train_loss_list = []
        self.validation_loss_list = []
        self.validation_accuracy_list = []

    def train_seq2seq_model(self, questions, questions_tensor, equations_tensor, SOS_token_idx, device):
        self.adam_optimizer.zero_grad()

        question_tokens_length, contextualized_question_tokens = self.wordEmbedding.forward(questions, questions_tensor)
        sorted_question_tokens_length = sorted(question_tokens_length, key=int, reverse=True)
        sorted_question_token_indexes = torch.LongTensor(sorted(range(len(question_tokens_length)), key=lambda i: question_tokens_length[i], reverse=True)).to(device)
        question_token_indexes = torch.LongTensor(sorted(range(len(question_tokens_length)), key=lambda i: sorted_question_token_indexes[i])).to(device)
        sorted_question_tensor = torch.index_select(input=contextualized_question_tokens, dim=1, index=sorted_question_token_indexes)
        encoder_output, encoder_hidden = self.encoder.forward(sorted_question_tensor, sorted_question_tokens_length, question_token_indexes)

        decoder_input = torch.tensor([SOS_token_idx for i in range(0, len(equations_tensor[0]))], device=device)

        if self.param.gru:
            decoder_hidden = encoder_hidden[:self.param.num_layers]
        else:
            decoder_hidden = (encoder_hidden[0][:self.param.num_layers], encoder_hidden[1][:self.param.num_layers])

        decoder_output, decoder_hidden = self.decoder.forward(encoder_output, decoder_input, decoder_hidden)
        loss = self.NLLLoss(decoder_output, equations_tensor[0])

        for i in range(1, len(equations_tensor)):
            decoder_output, decoder_hidden = self.decoder.forward(encoder_output, decoder_input=equations_tensor[i-1], last_hidden=decoder_hidden)
            loss += self.NLLLoss(decoder_output, equations_tensor[i])

        loss.backward()

        torch.nn.utils.clip_grad_norm_(parameters=self.parameters, max_norm=param.max_norm)
        self.adam_optimizer.step()

        return loss.item() / len(equations_tensor)

    def validate(self, dev_data, train_word_voc, train_equation_voc, SOS_token_idx, EOS_token_idx, device):
        with torch.no_grad():
            number_of_equations = 0
            number_of_correct_equations = 0
            self.validation_loss = 0
            bleu_score = 0

            print("\n\n\t\t--- VALIDATION ---")
            for batch_data in dev_data:
                print("\n\t\tNew Validation Batch")
                questions_tensor, equations_tensor = data_processing.find_data_in_vocabulary(train_word_voc, train_equation_voc, batch_data["Question"], batch_data["Equation"], device)
                question_tokens_length, contextualized_question_tokens = self.wordEmbedding.forward(batch_data["Question"], questions_tensor)

                sorted_question_tokens_length = sorted(question_tokens_length, key=int, reverse=True)
                sorted_question_token_indexes = torch.LongTensor(sorted(range(len(question_tokens_length)), key=lambda i: question_tokens_length[i], reverse=True)).to(device)
                question_token_indexes = torch.LongTensor(sorted(range(len(question_tokens_length)), key=lambda i: sorted_question_token_indexes[i])).to(device)
                sorted_question_tensor = torch.index_select(input=contextualized_question_tokens, dim=1,index=sorted_question_token_indexes)

                encoder_output, encoder_hidden = self.encoder.forward(sorted_question_tensor, sorted_question_tokens_length, question_token_indexes)

                decoder_input = torch.tensor([SOS_token_idx for i in range(0, len(equations_tensor[0]))], device=device)

                if self.param.gru:
                    decoder_hidden = encoder_hidden[:self.param.num_layers]
                else:
                    decoder_hidden = (encoder_hidden[0][:self.param.num_layers], encoder_hidden[1][:self.param.num_layers])

                val_loss = 0
                generated_equations = [[] for i in range(0, len(equations_tensor[0]))]

                for i in range(0, len(equations_tensor)):
                    decoder_output, decoder_hidden = self.decoder.forward(encoder_output, decoder_input=decoder_input, last_hidden=decoder_hidden)
                    val_loss += self.NLLLoss(decoder_output, equations_tensor[i])

                    values, indices = torch.topk(input=decoder_output, k=1)

                    for j in range(0, len(equations_tensor[0])):
                        if indices[j].item() == EOS_token_idx:
                            continue
                        generated_equations[j].append(train_equation_voc.get_by_id(indices[j].item()))
                    decoder_input = indices.squeeze().detach()

                decoder_answers, number_of_matches = data_processing.solve_equation(generated_equations, batch_data["Numbers"], batch_data["Answer"])
                number_of_correct_equations += number_of_matches
                number_of_equations += len(decoder_answers)
                bleu_score += (calculate_bleu_score(batch_data["Equation"], generated_equations) / len(equations_tensor))
                actual_and_predicted_equation_comparison(batch_data["Equation"], generated_equations)
                self.validation_loss += val_loss / len(equations_tensor)

            validation_accuracy = number_of_correct_equations / number_of_equations
            self.validation_loss /= len(dev_data)
            bleu_score /= len(dev_data)

            return self.validation_loss, validation_accuracy, bleu_score

def save_model_weights(model, epoch, train_loss_list, validation_loss_list, validation_accuracy_list):
    state = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'optimizer_state_dict': model.adam_optimizer.state_dict(),
        'train_loss_list': train_loss_list,
        'validation_loss_list': validation_loss_list,
        'validation_accuracy_list': validation_accuracy_list
    }
    model_path = "model_weights\\model.pth".replace("\\", os.sep)
    torch.save(state, model_path)
    print("\tModel parameters are saved.\n")

def load_model_parameters(wordVoc, equationVoc):
    model_path = "model_weights\\model.pth".replace("\\", os.sep)
    checkpoint = torch.load(model_path)

    model = Seq2Seq(wordVoc, equationVoc)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.epoch = checkpoint["epoch"]
    model.adam_optimizer = checkpoint["optimizer_state_dict"]
    model.train_loss_list = checkpoint["train_loss_list"]
    model.validation_loss_list = checkpoint["validation_loss_list"]
    model.validation_accuracy_list = checkpoint["validation_accuracy_list"]
    return model

def save_loss_and_acc(string):
    print(string)
    txt_path = "model_weights\\loss_and_accuracy.txt".replace("\\", os.sep)
    file = open(txt_path, "a")
    file.write(string)
    file.close()

def train(seq2seqModel, wordVoc, equationVoc, train_data, dev_data, device):
    train_loss_list = []
    validation_loss_list = []
    validation_accuracy_list = []

    for epoch in range(0, param.epoch):
        print("\tEPOCH {}:".format(epoch+1))
        batch = 1
        train_loss = 0
        SOS_token_idx = equationVoc.get_by_operator("<s>")
        EOS_token_idx = equationVoc.get_by_operator("</s>")

        for batch_data in train_data:
            print("\n\t\tBatch {}/{} in Epoch {}:".format(batch, len(train_data), epoch + 1))
            questions_tensor, equations_tensor = data_processing.find_data_in_vocabulary(wordVoc, equationVoc, batch_data["Question"], batch_data["Equation"], device)
            seq2seqModel.train()
            train_loss += seq2seqModel.train_seq2seq_model(questions=batch_data["Question"], questions_tensor=questions_tensor, equations_tensor=equations_tensor, SOS_token_idx=SOS_token_idx, device=device)
            batch += 1

        train_loss = round(train_loss/len(train_data), 3)
        train_loss_list.append(train_loss)

        seq2seqModel.eval()
        validation_loss, validation_accuracy, bleu_score = seq2seqModel.validate(dev_data, wordVoc, equationVoc, SOS_token_idx=SOS_token_idx, EOS_token_idx=EOS_token_idx, device=device)
        validation_loss_list.append(validation_loss)
        validation_accuracy_list.append(validation_accuracy)
        save_loss_and_acc("\n\tEPOCH {} - Train loss: {}\tValidation loss: {}\tValidation accuracy: {}\tAverage BLEU-4 Score: {}".format(epoch+1, train_loss, validation_loss, validation_accuracy, bleu_score))
        print("\n\tEPOCH {} COMPLETED\n".format(epoch+1))

        # save_model_weights(seq2seqModel, epoch+1, train_loss_list, validation_loss_list, validation_accuracy_list)

def calculate_bleu_score(real_equations, generated_equations):
    score = 0
    for i in range(0, len(generated_equations)):
        score += sentence_bleu([list(real_equations[i].split())], generated_equations[i], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method4)
    return score

def actual_and_predicted_equation_comparison(real_equations, generated_equations):
    for i in range(0, len(generated_equations)):
        print("**** ", real_equations[i], generated_equations[i])
        if real_equations[i] == " ".join(generated_equations[i]):
            if real_equations[i] in actual_predicted_equations_voc:
                actual_predicted_equations_voc[real_equations[i]] += 1
            else:
                actual_predicted_equations_voc[real_equations[i]] = 1
    print(actual_predicted_equations_voc)