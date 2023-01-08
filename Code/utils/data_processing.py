import os
import re
import json
import torch
import pickle
import warnings
import numpy as np
from random import shuffle
from torch.autograd import Variable
from torch.utils.data import DataLoader
warnings.simplefilter(action='ignore', category=FutureWarning)
import utils.param as param
param = param.param()

def read_data(path):
    with open(path, encoding="utf8") as f:
        data = json.load(f)
    data = shuffle_and_split(data)
    return data

def shuffle_and_split(data):
    return DataLoader(data, batch_size=param.batch_size, shuffle=param.shuffle, num_workers=param.num_workers)

class EquationVoc:
    def __init__(self):
        self.operators_voc = {'<unk>': 0, '<s>': 1, '</s>': 2}
        self.operator_ids_voc = {0: '<unk>', 1: '<s>', 2: '</s>'}
        self.operators_by_count = {'<unk>': 1, '<s>': 1, '</s>': 1}
        self.number_of_distinct_operators = 3

    def create_equation_vocabulary(self, dataloaders):
        for dataloader in dataloaders:
            for batch in dataloader:
                for equation in batch["Equation"]:
                    operators = equation.split(" ")
                    for operator in operators:
                        if operator not in self.operators_voc:
                            self.operators_voc[operator] = self.number_of_distinct_operators
                            self.operator_ids_voc[self.number_of_distinct_operators] = operator
                            self.number_of_distinct_operators += 1
                            self.operators_by_count[operator] = 1
                        else:
                            self.operators_by_count[operator] += 1

    def get_by_id(self, id):
        return self.operator_ids_voc[id]

    def get_by_operator(self, operator):
        return self.operators_voc[operator]

class WordVoc:
    def __init__(self):
        self.words_voc = {'<unk>': 0, '<s>': 1, '</s>': 2}
        self.word_ids_voc = {0: '<unk>', 1: '<s>', 2: '</s>'}
        self.words_by_count = {'<unk>': 1, '<s>': 1, '</s>': 1}
        self.number_of_distinct_words = 3

    def create_word_vocabulary(self, dataloaders):
        for dataloader in dataloaders:
            for batch in dataloader:
                for question in batch["Question"]:
                    question_words = question.split(" ")
                    for word in question_words:
                        if word not in self.words_voc:
                            self.words_voc[word] = self.number_of_distinct_words
                            self.word_ids_voc[self.number_of_distinct_words] = word
                            self.number_of_distinct_words += 1
                            self.words_by_count[word] = 1
                        else:
                            self.words_by_count[word] += 1

    def get_by_id(self, id):
        return self.word_ids_voc[id]

    def get_by_word(self, word):
        return self.words_voc[word]

def save_dict_as_pkl(output_path, vocabulary):
    with open(output_path, 'wb') as f:
        pickle.dump(vocabulary, f, protocol=pickle.HIGHEST_PROTOCOL)

def read_pkl_file(path):
    with open(path, 'rb') as handle:
        pkl = pickle.load(handle)
    return pkl

def find_data_in_vocabulary(wordVoc, equationVoc, questionBatch, equationBatch, device):
    one_hot_encoded_questions = []
    one_hot_encoded_equations = []

    end_of_question_tag = wordVoc.get_by_word("</s>")
    end_of_equation_tag = equationVoc.get_by_operator("</s>")

    for data in questionBatch:
        one_hot_encoded_question = []
        for d in data.split(" "):
            one_hot_encoded_question.append(wordVoc.get_by_word(d))

        one_hot_encoded_question.append(end_of_question_tag)
        one_hot_encoded_questions.append(one_hot_encoded_question)

    for data in equationBatch:
        one_hot_encoded_equation = []
        for e in data.split(" "):
            one_hot_encoded_equation.append(equationVoc.get_by_operator(e))

        one_hot_encoded_equation.append(end_of_equation_tag)
        one_hot_encoded_equations.append(one_hot_encoded_equation)

    return create_tensor(one_hot_encoded_questions, end_of_question_tag).to(device), create_tensor(one_hot_encoded_equations, end_of_equation_tag).to(device)

def create_tensor(encoded_batch, end_tag_index):
    maxLength = max([len(sub_item) for sub_item in encoded_batch])

    for sub_item in encoded_batch:
        sub_item += [end_tag_index for i in range(0, maxLength - len(sub_item))]

    return Variable(torch.LongTensor(np.transpose(encoded_batch)))

def unmask_equations(eq, nums):
    for i in range(0, len(eq)):
        numbers = nums[i].split(" ")
        for n in range(0, len(numbers)):
            eq[i] = eq[i].replace("number" + str(n), numbers[n])
    return eq

def solve_equation(equation, numbers, answers):
    equation = [*map(" ".join, equation)]
    equation = unmask_equations(equation, numbers)
    operations = ["-", "+", "*", "/"]

    for j in range(0, len(equation)):
        try:
            eq = equation[j].split(" ")
            operators = [i for i, s in enumerate(eq) if len(re.findall('[0-9]', s)) == 0]

            while len(operators) > 0 and len(eq) > 1:
                op = eq[operators[-1]]
                e = "".join((eq[operators[-1]], eq.pop(operators[-1] + 2)))
                e = "".join((eq.pop(operators[-1] + 1), e))

                if op == operations[0]:
                    ans = int(e.split(op)[0]) - int(e.split(op)[1])
                if op == operations[1]:
                    ans = int(e.split(op)[0]) + int(e.split(op)[1])
                if op == operations[2]:
                    ans = int(e.split(op)[0]) * int(e.split(op)[1])
                if op == operations[3]:
                    ans = int(int(e.split(op)[0]) / int(e.split(op)[1]))

                eq[operators[-1]] = str(ans)
                if len(eq) > 1:
                    operators = [i for i, s in enumerate(eq) if len(s) == 1 and len(re.findall('[0-9]', s)) == 0]
            equation[j] = eq[0]
        except:
            equation[j] = ""

    number_of_matches = find_matches(equation, answers)
    return equation, number_of_matches

def find_matches(decoder_answers, real_answers):
    correct = [y for x, y in enumerate(real_answers) if y == decoder_answers[x]]
    return len(correct)

def random_and_split_dataset(json_data, path, size):
    shuffle(json_data)

    train = json_data[0:size]
    dev = json_data[size:]

    with open(os.path.join(path, "translated_train_data.json"), "w", encoding='utf-8') as f:
        json.dump(train, f)
    f.close()

    with open(os.path.join(path, "translated_dev_data.json"), "w", encoding='utf-8') as f:
        json.dump(dev, f)
    f.close()