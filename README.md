# Turkish MWP Corpora

Math word problem (MWP) solving is a challenging task since the semantic gap between natural language texts and mathematical equations. The main purpose of the task is to receive a textual math problem as input and generate an appropriate equation for solving that problem.

To solve elementary-level math problems, we introduce new Turkish MWP corpora, by translating and combining English benchmark datasets, which are  [MAWPS](https://github.com/sroy9/mawps), [ASDiv-A](https://github.com/chaochun/nlu-asdiv-dataset), [SVAMP](https://github.com/arkilpatel/SVAMP), and [MathQA](https://math-qa.github.io/). After manual arrangements and preprocessing, we publish the corpora consisting of question texts, equations, and answers customized to our model.

## Corpora

We generated two distinct corpora which differ in their complexity levels. The details of the relevant datasets are as follows.

### Combined Dataset from MAWPS, ASDiv-A, and SVAMP
MAWPS is a frequently used English benchmark dataset containing equation templates and 3320 questions. ASDiv-A is a diverse corpus in terms of lexicon patterns and problem types with 1218 data. SVAMP is a challenging dataset with 1000 data samples and injected several types of modifications into a set of seed problems derived from the ASDiv-A dataset.

In Turkish version, these three datasets are chosen to merge. In total, 4163 MWP data are provided by adding a few manual questions.

There are 862 data samples in the dev set and 3301 samples in the training set.

### MathQA Dataset
The MathQA benchmark dataset consisting of 37200 data is employed as the second dataset. It is one of the most challenging datasets, the amount of data is satisfactory, and it covers a variety of questions from many aspects.

After visual inspections, the dataset is reduced to 19555 data samples in Turkish version. Physics, geometry, some of the probability, economics, and interest problems that require knowledge of formulas and equations with many unknowns are eliminated.

There are 3904 data samples in the dev set and 15651 samples in the training set.

## Format
We constructed the final version of the corpora in JSON format, including the ``Question``, ``Equation``, ``Numbers``, and ``Answer`` fields.

- ``Question``: This field includes written math problem. All words are lowercase. There is a space before and after punctuation marks. The numbers in the questions are replaced with the numberX tags, where X is the rank of that number among all the numbers in the sentence.

- ``Equation``: It corresponds to the formula that provides the mathematical solution of the written math problem. A common template structure is applied to the equations in the dataset. The operator to be applied to two operands is declared just before these operands. This type of notation is called the prefix notation, e.g. "(- number0 number1)".

- ``Numbers``: The list of numbers in the written math problem according to their orders.

- ``Answer``: The annodated answer of the written math problem.

## License
This Turkish MWP Corpora is made available under the [Open Data Commons Attribution License](http://opendatacommons.org/licenses/by/1.0/). This is a human-readable summary of the ODC-BY 1.0 license. Read the [LICENSE](LICENSE) file for details.

You are free:

- To share: To copy, distribute, cite and use the database.
- To create: To produce works from the database.
- To adapt: To modify, transform and build upon the database.

As long as you:

- Attribute: You must attribute any public use of the database, or works produced from the database, in the manner specified in the license. For any use or redistribution of the database, or works produced from it, you must make clear to others the license of the database and keep intact any notices on the original database.git add README.md.