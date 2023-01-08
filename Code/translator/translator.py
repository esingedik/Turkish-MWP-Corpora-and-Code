from googletrans import Translator
import numpy as np

def translate_to_turkish(df, path, question_list):
    print("Translation started for {} set.".format(df))
    x = 0
    while x < len(question_list):
        translator = Translator()
        translations = translator.translate(question_list[x: x+10], dest='tr')  # chunk_size = 300 due to translation restriction
        x += 10

        with open(path, "a", encoding="utf-8") as outfile:
            for translation in translations:
                outfile.write(translation.text + "\n")
        outfile.close()
    print("Translation is completed for {} group of questions".format((len(question_list)/300)))