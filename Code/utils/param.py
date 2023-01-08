class param():
    shuffle = True
    num_workers = 5
    batch_size = 8
    hidden_size = 256
    num_layers = 2
    dropout = (0 if num_layers == 1 else 0.1)
    uniform_dist_bound = 0.05
    max_norm = 1  # Clip gradients to max norm
    learning_rate = 2e-4
    epoch = 50
    gru = False
    embedding_dimension_for_equations = 16  # embedding dimension for attention decoder
    embedding_learning_rate = 8e-6  # for word embedding
    embedding_model = "BERT"  # BERT, ELECTRA, CONVBERT, Word2vec, Fasttext, GloVe
    bert_model = "dbmdz/bert-base-turkish-128k-uncased"
    electra_model = "dbmdz/electra-base-turkish-mc4-uncased-generator"
    convbert_model = "dbmdz/convbert-base-turkish-mc4-uncased"

    if embedding_model == "ELECTRA":
        input_embedding_size = 256

    elif embedding_model == "CONVBERT" or embedding_model == "BERT":
        input_embedding_size = 768  # 768-dimensional space from BERT

    elif embedding_model == "Word2vec":
        input_embedding_size = 400  # 400-dimensional space from Turkish Word2vec pre-trained model
        model_path = "model\\tr_word2vec_model"

    elif embedding_model == "Fasttext":
        input_embedding_size = 300  # 300-dimensional space from Turkish Fasttext pre-trained model
        model_path = "model\\tr_fasttext_model"

    elif embedding_model == "GloVe":
        input_embedding_size = 300  # 300-dimensional space from Turkish Glove pre-trained model
        model_path = "model\\tr_glove_model.txt"