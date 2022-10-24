# Semantic Similarity with Siamese neural network (Manhattan_LSTM)
Сопоставление названий компаний с помощью Сиамской нейронной сети

## Getting started

Requirements

* Python >= 3.7
* GPU >= 16GB
* CUDA >= 11.3

1) Очистка набора данных и приведение к формату .tsv:

````
$ python Datasets/clear.py
$ python Datasets/csv_to_tsv.py
````

2) Запуск обучения:

````
$ python PyTorch/main.py
````

3) Инференс:

````
$ python inf.py
````

## Dataset
В наборе данных 4 столбца и 497819 строк. Данные имеют слеедующий вид:

| pair_id |         name_1          |                name_2                | is_duplicate | 
|:-------:|:-----------------------:|:------------------------------------:|:------------:|
|    1    |   Iko Industries Ltd.   | Enormous Industrial Trade Pvt., Ltd. |      0       | 
|    2    | Pirelli Neumaticos SAIC |        Pirelli Tyre Co., Ltd.        |      1       | 
|   ...   |           ...           |                 ...                  |     ...      |

где pair_id - id пары, name_1 - первое название, name_2 - второе название, is_duplicate - являются ли названия из name_1 и name_2 дубликатами (0 - нет, 1 - да). Названия компаний 
преимущественно на английском языке, но также встречаются другие языки (русский и турецкий).

Распределение по категориям в наборе данных неравномерно.

| is_duplicate |    Count     |
|:------------:|:------------:|
|      0       |    494161    |
|      1       |     3658     |

Данные можно загрузить по [ссылке](https://drive.google.com/file/d/1e9bdr7wcQX_YBudQcsKj-sMoIGxQOlK4/view?usp=sharing).

## Model
Для решения задачи определения сходство между парами названий была выбрана архитектура сиамской нейронной сети 
Manhattan_LSTM. В модели есть две идентичные сети LSTM. LSTM передает векторные представления предложений и выводит 
скрытое состояние, кодирующее семантическое значение предложений. Впоследствии эти скрытые состояния сравниваются с и
спользованием механизма сходства для вывода оценки сходства.

Первоначальные эмбеддинги для слов берутся из GoogleNews-vectors-negative300.bin.gz, загрузить можно по
[ссылке](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g).

Параметры модели:

|      Params       |       |
|:-----------------:|:-----:|
|    Hidden Size    |  50   |
|    Batch Size     |  32   |
| Max. input length |  20   |
|   Learning rate   | 0.001 |
| Number of epochs  |  100  |

Обученную модель можно загрузить по [ссылке](). 
Результаты валидации на тестовой выборке представлены в таблице.

|     Model      | Accuracy  | Loss | Precision | Recall  |
|:--------------:|:---------:|:----:|:---------:|:-------:|
| Manhattan_LSTM |   99.86   | 3.90 |   94.59   |  85.36  |


График потерь:

[img]()


## Comparison 
Для оценки качества работы обученной сиамской нейронной сети результаты ее работы сравнили с обработкой NLP библиотеки
SpaCy.

|     Model      | Accuracy | Precision | Recall |
|:--------------:|:--------:|:---------:|:------:|
| Manhattan_LSTM |  99.86   |   94.59   | 85.36  |
|     SpaCy      |   ...    |    ...    |  ...   |