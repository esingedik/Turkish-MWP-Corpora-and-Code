import re
import json
from decimal import *

def mask(df, output_path, data, questions, output_extension):
    for i in range(0, len(data)):
        data[i]["Answer"], data[i]["Numbers"], data[i]["Equation"], questions[i] = arrange_numbers(data[i]["Answer"], data[i]["Numbers"], data[i]["Equation"], questions[i])
        numbers = data[i]["Numbers"].split(" ")
        questions[i] = arrange_punctuations(questions[i].lower())
        for n in range(0, len(numbers)):
            questions[i] = questions[i].replace(numbers[n] + " ", "number" + str(n) + "_ ")
        questions[i] = re.sub('([.!?])', r' \1', questions[i])  # add space before other punctuations
        questions[i] = questions[i].replace("_", "")            # remove underscore which is used for masking the numbers
        data[i]["Question"] = questions[i]
    if output_extension == "json":
        save_to_json_file(df, output_path, data)
    else:
        save_to_txt_file(df, output_path, data)

def arrange_numbers(answer, numbers, equation, question):
    remove_trailing_zeroes = re.compile('(?<![\d.])(?![1-9]\d*(?![\d.])|\d*\.\d*\.)0*(?!(?<=0)\.)([\d.]+?)\.?0*(?![\d.])')
    answer = remove_trailing_zeroes.sub('\\1', str(answer))  # decimals in integers are removed such as "52.0" -> "52"
    numbers = remove_trailing_zeroes.sub('\\1', numbers)
    equation = remove_trailing_zeroes.sub('\\1', equation)
    question = remove_trailing_zeroes.sub('\\1', question)
    question = question.replace(",0", "")
    return answer, numbers, equation, question

def arrange_punctuations(question):
    question = re.sub(r'[,]+(?![0-9])', r' ,', question)        # add space before the comma only if not followed by a number
    question = re.sub('([\'])+(?![\s])', r' \1 ', question)     # add space before and after the single quotes
    question = re.sub('([$])', r' \1', question)                # add space before the dollar sign
    question = re.sub('([%])', r'\1 ', question)                # add space after the percent sign
    question = re.sub(r'[,]+([0-9])', r'.\1', question)         # change comma with dot for the decimals
    question = question.replace("\n", "")
    return question

def save_to_txt_file(df, output_path, data):
    with open(output_path, "w", encoding="utf-8") as outfile:
        outfile.write(str(data))
    print("Translated {} data is saved to the txt file.".format(df))

def save_to_json_file(df, output_path, data):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    f.close()
    print("Translated {} data is saved to the json file.".format(df))
