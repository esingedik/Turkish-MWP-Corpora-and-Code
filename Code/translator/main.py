import preprocess_data as preprocessor
import translator as translator
import postprocess_data as postprocessor
import json
import re

dataset = "train"  # train or dev
extension = "txt"  # txt or json
parent_data_path = "..\\..\\Data\\english_data\\MAWPS-ASDIV-SWAMP"

def main():
    data, question_list = preprocessor.extract(df=dataset, parent_data_path=parent_data_path)
    ## question_list = translator.translate_to_turkish(df=dataset, path="{}\\turkish_data\\translated_{}_data.txt".format(parent_data_path, dataset), question_list=question_list)

    with open(parent_data_path + "\\turkish_data\\translated_{}_questions.txt".format(dataset), encoding="utf8") as f:
        question_list = f.readlines()
    postprocessor.mask(df=dataset, output_path="{}\\turkish_data\\translated_{}_data.{}".format(parent_data_path, dataset, extension), data=data, questions=question_list, output_extension=extension)

if __name__ == '__main__':
    main()