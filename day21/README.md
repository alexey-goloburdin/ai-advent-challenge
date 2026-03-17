Пайплайн для индексации локальных документов с генерацией эмбеддингов через [Ollama](https://ollama.com/) (`nomic-embed-text`). Реализует две стратегии chunking и сохраняет индекс в JSON с метаданными.


## Возможности

- Загрузка документов форматов `.md`, `.txt`, `.py`, `.rst`, `.pdf`
- Две стратегии разбиения на чанки:
  - **Fixed-size** — фиксированный размер с перекрытием (overlap)
  - **Structural** — по структуре документа (заголовки, функции, абзацы)
- Генерация эмбеддингов через Ollama (`nomic-embed-text`)
- Сохранение индекса в JSON с полными метаданными
- Сравнение стратегий: статистика, примеры чанков

## Структура проекта


```
day21/
├── documents/          # исходные документы (положи сюда свои файлы)
│   ├── *.md
│   ├── *.py
│   └── *.pdf
├── index/              # сгенерированные индексы (создаётся автоматически)
│   ├── fixed.json
│   └── structural.json
├── src/
│   ├── loader.py       # загрузка документов → plain text
│   ├── chunkers.py     # две стратегии chunking
│   ├── embedder.py     # генерация эмбеддингов через Ollama API
│   ├── indexer.py      # сохранение / загрузка индекса
│   └── compare.py      # сравнение стратегий
├── main.py             # точка входа
└── README.md
```

## Требования

- Python 3.10+
- [Ollama](https://ollama.com/) запущен локально (`http://localhost:11434`)
- Модель `nomic-embed-text` загружена в Ollama


Опционально — для работы с PDF:

```bash
pip install pypdf

```

## Установка

```bash
# Клонировать / скачать проект
cd day21

# Установить зависимости (только для PDF)
pip install pypdf


# Загрузить модель в Ollama
ollama pull nomic-embed-text
```


## Использование


### Базовый запуск


Положи документы в папку `documents/` и запусти:

```bash
python main.py
```

Будут применены обе стратегии, выведено сравнение, сгенерированы и сохранены эмбеддинги.

### Аргументы командной строки

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--docs` | `documents` | Папка с исходными документами |
| `--index-dir` | `index` | Папка для сохранения индексов |
| `--chunk-size` | `500` | Размер чанка в символах (для fixed) |

| `--overlap` | `50` | Перекрытие между чанками (для fixed) |
| `--model` | `nomic-embed-text` | Модель Ollama для эмбеддингов |

| `--strategy` | `both` | Стратегия: `fixed` / `structural` / `both` |
| `--no-embed` | `False` | Только chunking, без генерации эмбеддингов |


### Примеры

```bash
# Только посмотреть chunking, без обращения к Ollama
python main.py --no-embed

# Только fixed-size стратегия, кастомный размер чанка
python main.py --strategy fixed --chunk-size 800 --overlap 100


# Только structural стратегия
python main.py --strategy structural

# Другая папка с документами
python main.py --docs ~/my-notes --index-dir ~/my-index
```

## Формат индекса

Каждый файл индекса (`fixed.json`, `structural.json`) — массив записей:

```json
{
  "chunk_id":    "langchain_docs.md::structural::3",
  "text":        "## Installation\n\npip install langchain...",

  "source":      "langchain_docs.md",
  "title":       "langchain_docs",
  "section":     "## Installation",
  "strategy":    "structural",
  "chunk_index": 3,
  "total_chunks": 18,
  "embedding":   [0.023, -0.117, ...]
}
```

### Поля метаданных

| Поле | Описание |
|---|---|
| `chunk_id` | Уникальный идентификатор: `source::strategy::index` |

| `text` | Текст чанка |
| `source` | Путь к файлу-источнику (относительно `documents/`) |
| `title` | Имя файла без расширения |
| `section` | Раздел/заголовок (только для structural; пусто для fixed) |
| `strategy` | `fixed` или `structural` |
| `chunk_index` | Порядковый номер чанка в документе |

| `total_chunks` | Всего чанков из этого документа |
| `embedding` | Вектор эмбеддинга (768 измерений для nomic-embed-text) |

## Стратегии chunking

### Fixed-size

Текст делится на отрезки фиксированного размера с перекрытием:

```
Текст:   [----chunk 1----][----chunk 2----][----chunk 3----]
                     [overlap]        [overlap]

```

**Плюсы:** предсказуемый размер чанков, простая реализация  
**Минусы:** может разрезать предложение или абзац посередине

### Structural

Текст делится по логическим границам, которые определяются по типу файла:

| Тип файла | Граница раздела |
|---|---|
| `.md` | Заголовки `#`, `##`, `###` |
| `.py` | Определения `class`, `def` |
| `.rst` | Подчёркивание `===`, `---` |
| `.txt` | Пустые строки |

Слишком большие разделы автоматически дробятся по абзацам.

**Плюсы:** семантически целостные чанки, наличие section-метаданных  
**Минусы:** непредсказуемый размер, документы без структуры дают один большой чанк


## Выбор размера чанка

Контекстное окно `nomic-embed-text` — 8192 токена, но размер чанка выбирается из соображений качества поиска, а не лимита модели:

| Размер | Характеристика |

|---|---|
| < 100 токенов | Мало контекста, эмбеддинг ненадёжен |
| 256–512 токенов | ✅ Оптимум для большинства задач |
| > 1000 токенов | Смысл "размывается", в LLM попадает шум |

Для символьного `chunk_size`: ~500 символов ≈ 128 токенов для английского / ~250 токенов для русского.

## Поддерживаемые форматы

| Формат | Загрузка |
|---|---|
| `.md`, `.txt`, `.rst` | Встроенная |
| `.py` | Встроенная |
| `.pdf` | Требует `pip install pypdf` |

## Следующий шаг

Готовый индекс можно использовать для семантического поиска — основы RAG-пайплайна:

```python
from src.indexer import load_index
from src.embedder import get_embedding
import numpy as np

index = load_index("index/structural.json")
query_vec = np.array(get_embedding("как установить зависимости?"))

scores = []
for record in index:
    vec = np.array(record["embedding"])
    score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))
    scores.append((score, record))


top = sorted(scores, reverse=True)[:3]
for score, chunk in top:
    print(f"{score:.3f} | {chunk['source']} | {chunk['section']}")
    print(chunk['text'][:200])
```
