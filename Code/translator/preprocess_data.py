
import pandas as pd
import json
import ast
import re

def read_csv(data_path):
    file_df = pd.read_csv(data_path)
    data = zip(file_df['Question'].values, file_df['Equation'].values, file_df['Numbers'].values, file_df['Answer'].values)
    return sorted(data, key=lambda x: len(x[0].split()))

def convert_to_json(data):
    data = [{"Question": d[0], "Equation": d[1],
             "Numbers": d[2], "Answer": d[3]}
            for d in data]
    return json.loads(json.dumps(data, indent=2))

def unmask_questions(json_data):
    for j in json_data:
        numbers = j["Numbers"].split(" ")
        for n in range(0, len(numbers)):
            j["Question"] = j["Question"].replace("number" + str(n), numbers[n])
    return json_data

def unmask_equations(json_data):
    for j in json_data:
        numbers = j["Numbers"].split(" ")
        for n in range(0, len(numbers)):
            j["Equation"] = j["Equation"].replace("number" + str(n), numbers[n])
    return json_data

def combine_equation(json_data):
    for data in json_data:
        eq = data["Equation"].split(" ")
        operators = [i for i, s in enumerate(eq) if len(re.findall('[0-9]', s)) == 0]

        while len(operators) > 0 and len(eq) > 1:
            e = "".join((eq[operators[-1]], eq.pop(operators[-1] + 2), ")"))
            e = "".join(("(", eq.pop(operators[-1] + 1), e))
            eq[operators[-1]] = e
            if len(eq) > 1:
                operators = [i for i, s in enumerate(eq) if len(s) == 1 and len(re.findall('[0-9]', s)) == 0]
        data["Equation"] = eq[0]
    return json_data

def save_the_all_data(data_path, df, data):
    with open(data_path, "w") as f:
        json.dump(data, f)
    f.close()
    print("English {} data is extracted and saved to the json file.".format(df))

def save_the_questions(data_path, df, data):
    questions = []
    with open(data_path, "w", encoding="utf-8") as outfile:
        # outfile.write("\n".join(re.sub(r'\s([?.!%/,\'’:=)"](?:\s|$))', r'\1', str(item)) for item in data))
        for i in data:
            outfile.write(str(i["Question"] + "\n"))
            questions.append(str(i["Question"]))
    print("English {} questions are extracted and saved to the txt file.".format(df))
    return questions

def regex_for_questions(data):
    for i in data:
        i["Question"] = re.sub("mrs.", "mrs", i["Question"])
        i["Question"] = re.sub("mr.", "mr", i["Question"])
        i["Question"] = re.sub(" n\'t", "n't", i["Question"])
        i["Question"] = re.sub("\’ s", "\'s", i["Question"])
        i["Question"] = re.sub(" \'s", "\'s", i["Question"])
        i["Question"] = re.sub(" - ", "-", i["Question"])
        i["Question"] = re.sub("\'ve", "'ve", i["Question"])
        i["Question"] = re.sub("\'re", "'re", i["Question"])
        i["Question"] = re.sub("\'d", "'d", i["Question"])
        i["Question"] = re.sub("\'ll", "'ll", i["Question"])
    return data

def extract(df, parent_data_path):
    json_data = convert_to_json(read_csv(data_path="{}\\english_data\\{}.csv".format(parent_data_path, df)))
    json_data = unmask_questions(json_data)
    json_data = regex_for_questions(json_data)
    questions = save_the_questions(data_path="{}\\english_data\\{}_questions.txt".format(parent_data_path, df), df=df, data=json_data)
    return json_data, questions

    # INFO: Following operations are for arranging the equations. e.g., (- number0 number1) -> (number0 - number1)
    # json_data = unmask_equations(json_data)
    # json_data = combine_equation(json_data)
    # save_the_all_data(data_path="{}\\english_data\\arranged_{}_data.json".format(parent_data_path, df), df=df, data=json_data)