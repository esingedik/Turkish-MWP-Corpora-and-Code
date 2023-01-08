import os
import sys
import torch
import torch.nn as nn
import fasttext.util
from transformers import BertModel
from transformers import BertTokenizer
from gensim.models import KeyedVectors
from transformers import AutoModel, AutoTokenizer
import utils.param as param

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Embedding(nn.Module):
    def __init__(self, device, wordVoc):
        super(Embedding, self).__init__()

        self.device = device
        self.param = param.param()
        self.wordVoc = wordVoc

        if self.param.embedding_model == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained(self.param.bert_model, do_lower_case=True)
            self.layer = BertModel.from_pretrained(self.param.bert_model)

        elif self.param.embedding_model == "ELECTRA":
            self.tokenizer = AutoTokenizer.from_pretrained(self.param.electra_model, do_lower_case=True)
            self.layer = AutoModel.from_pretrained(self.param.electra_model)

        elif self.param.embedding_model == "CONVBERT":
            self.tokenizer = AutoTokenizer.from_pretrained(self.param.convbert_model, do_lower_case=True)
            self.layer = AutoModel.from_pretrained(self.param.convbert_model)

        else:
            self.input_embedding_size = self.param.input_embedding_size
            self.dropout = nn.Dropout(self.param.dropout)
            self.path = param.param().model_path.replace("\\", os.sep)

            if self.param.embedding_model == "Word2vec":
                self.layer = nn.Embedding.from_pretrained(torch.FloatTensor(self.gensim_embeddings(binary=True)))

            elif self.param.embedding_model == "GloVe":
                self.layer = nn.Embedding.from_pretrained(torch.FloatTensor(self.gensim_embeddings(binary=False)))

            elif self.param.embedding_model == "Fasttext":
                self.layer = nn.Embedding.from_pretrained(torch.FloatTensor(self.fasttext_embeddings()))

        print("\tInitializing Turkish {} Embedding Model...".format(self.param.embedding_model))

    def gensim_embeddings(self, binary=False):
        word2vec_vector = KeyedVectors.load_word2vec_format(self.path, binary=binary)
        tensor_array = torch.randn(self.wordVoc.number_of_distinct_words, self.input_embedding_size)

        for i in range(0, self.wordVoc.number_of_distinct_words):
            word = self.wordVoc.get_by_id(i)
            if word in word2vec_vector:
                tensor_array[i] = torch.tensor(word2vec_vector[word])
            else:
                tensor_array[i] = ((torch.rand(size=(1, self.input_embedding_size)) < 0.25).int()).clone().detach()
        return tensor_array

    def fasttext_embeddings(self):
        fasttext_vector = fasttext.load_model(self.path)
        tensor_array = torch.randn(self.wordVoc.number_of_distinct_words, self.input_embedding_size)

        for i in range(0, self.wordVoc.number_of_distinct_words):
            word = self.wordVoc.get_by_id(i)
            tensor_array[i] = torch.tensor(fasttext_vector.get_word_vector(word))
        return tensor_array

    def forward(self, questions, questions_tensor):
        print("\t\t\tRunning {} Model...".format(self.param.embedding_model))

        if self.param.embedding_model == "Word2vec" or self.param.embedding_model == "Fasttext" or self.param.embedding_model == "GloVe":
            question_tokens_length = []
            for question in questions:
                question_tokens_length.append(len(question.split(" ")))

            embedded = self.dropout(self.layer(questions_tensor))
            return question_tokens_length, embedded

        else:
            question_tokens = []
            question_tokens_length = []
            max_tokens_length = 0

            for question in questions:
                question_tokens.append(['[CLS]'] + self.tokenizer.tokenize(question) + ['[SEP]'])
                question_tokens_length.append(len(question_tokens[-1]))
                if len(question_tokens[-1]) > max_tokens_length:
                    max_tokens_length = len(question_tokens[-1])

            for token in question_tokens:
                token += ['[PAD]' for t in range(0, max_tokens_length - len(token))]

            padding_token = self.tokenizer.convert_tokens_to_ids(tokens='[PAD]')
            question_token_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens) for tokens in question_tokens], device=self.device)

            attention_mask = (question_token_ids != padding_token).long()

            outputs = self.layer(input_ids=question_token_ids, attention_mask=attention_mask)
            contextualized_tokens = outputs[0].transpose(0, 1).to(self.device)
            return question_tokens_length, contextualized_tokens