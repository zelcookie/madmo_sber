
# Задача
Соревнование на Kaggle [Tweet Sentiment Extraction](https://www.kaggle.com/c/tweet-sentiment-extraction)

Необходимо по тексту твита и его тональности выделить слово или фразу отражающую указанную тональность.

Для оценки качества предсказаний используется мера Жаккара

Задача наиболее похожа на SQuAD[3,4], где необходимо по вопросу и фрагменту текста найти в нем ответ. В данной задаче отсутствует вопрос, но есть разметка тональности.



# Результаты
| Model                                       | Val Score     | Public Score  | Private Score |
| ------------------------------------------- |:-------------:|:-------------:|--------------:|
| bert_baseline                               | 0.649         | 0.651         |  0.649        |
| bert_with_sentiment                         | 0.704         | 0.706         |  0.706        |
| bert_with_sentiment_cv                      | 0.701         | 0.700         |  0.707        |
| roberta_with_sentiment                      | 0.712         | 0.705         |  0.708        |
| bert_with_sentiment_es                      | 0.696         | 0.702         |  0.705        | 
| roberta_with_sentiment_es                   | 0.710         | 0.706         |  0.704        |
| bert_with_sentiment_cv_es                   | 0.700         | 0.700         |  0.704        |
| roberta_with_sentiment_cv                   | 0.711         | 0.709         |  0.714        |
| roberta_with_sentiment_cv_es                | 0.710         | 0.706         |  0.713        |
| roberta_with_sentiment_cv_5_ep              | 0.714         | 0.709         |  0.713        |
| roberta_large_with_sentiment                | 0.705         | 0.700         |  0.698        |
| roberta_large_with_sentiment_cv             | 0.704         | 0.705         |  0.705        |
| bert_with_sentiment_CNN_1                   | 0.708         | 0.702         |  0.703        |
| bert_with_sentiment_CNN_2                   | 0.698         | 0.696         |  0.700        |
| roberta_large_with_sentiment_CNN_2_cv       | 0.705         | 0.700         |  0.703        |
| roberta_with_sentiment_CNN_1_cv             | 0.712         | 0.709         |  0.712        |
| roberta_with_sentiment_CNN_2_cv             | 0.711         | 0.705         |  0.710        |
| conversational_bert_with_sentiment          | 0.709         | 0.704         |  0.706        |
| roberta_with_sentiment_cv_10                | 0.719         | 0.713         |  0.715        |
| roberta_with_sentiment_cv_10_2_l            | 0.714         | 0.711         |  0.715        |
| roberta_with_sentiment_cv_10_no_neutral     | 0.711         | 0.709         |  0.714        |
| roberta_with_sentiment_cv_10_2_l_no_neutral | 0.712         | 0.701         |  0.704        |
| roberta_with_sentiment_cv_no_neutral        | 0.708         | 0.707         |  0.709        |

Во всех моделях предсказываем начало и конец целевой фразы как в [1] для SQuAD
## bert_baseline
BERT-Base[1] + полносвязный слой основано на [этом видео](https://www.youtube.com/watch?v=XaQ0CBlQ4cY). Данные о тональности не использовались

## bert_with_sentiment

BERT-Base[1] + полносвязный слой, в качестве первого предложения используется метка тональности. Основано на [этом примере](https://www.kaggle.com/abhishek/bert-base-uncased-using-pytorch)



## roberta_with_sentiment

RoBerta-base[2] + полносвязный слой(на вход подавались выходы с двух последних слоев RoBerta) в качестве первого предложения используется метка тональности.  Основано на [этом примере](https://www.kaggle.com/abhishek/roberta-inference-5-folds)

## roberta_large_with_sentiment

Замена RoBerta-base на RoBerta-large[2]



## roberta_with_sentiment_cv_5_ep

 Уменьшино количество эпох 10 -> 5, было предположение, что сможет лучше сойтись, т.к. при 10 сходилось менее чем за 5, но при 5 в используемой конфигурации быстрее бы убывала скорость обучения



## X_cv

X натренированный на 5 фолдах, в качестве предсказания берется усреднее 5 полученных моделей

## X_es

X тренировался с loss early stopping вместо jaccard score early stopping

## X_CNN_y

Добавление, перед полносвязным слоем, *y* слоев одномерной свертки

## roberta_with_sentiment_cv_10

На вход полносвязному слою подавалось усреднение последних 4 слоев Bert, как [тут](https://www.kaggle.com/shoheiazuma/tweet-sentiment-roberta-pytorch). Обучалось 10 фолдов и усреднялось

## roberta_with_sentiment_cv_10_2_l

На вход полносвязному слою подавалась конкатенация сумм последних двух и 3его и 4го слоев Bert. Обучалось 10 фолдов и усреднялось


## X_no_neutral

При тренировке не использовались данные с меткой neutral, т.к. там как правило просто скопирован текст


# Выводы

Были обробованны различные модели, на данный момент лучше всего себя показала RoBerta-base с кроссвалидацией. К сожалению разметка не очень качественная, а отличие от других не такое большое, так что сложно сделать однозначные выводы. 


# Дальнейшие планы
 
 В дальнейшем планируется адаптировать под данную задачу и опробовать различные подходы из [5]


# Статьи

1. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
2. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)
3. [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/pdf/1606.05250.pdf)
4. [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/pdf/1806.03822.pdf)
5. [BERT for Question Answering on SQuAD 2.0](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15848021.pdf)
