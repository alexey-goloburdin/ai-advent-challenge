Перед использованием необходимо задать переменную окружения `OPENAI_BASE_URL` и `OPENAI_API_KEY`, например, так:

```shell
export OPENAI_BASE_URL="https://openai.api.proxyapi.ru/v1"
export OPENAI_API_KEY="..."
```

В `OPENAI_BASE_URL` можно указать любой URL для OpenAI-совместимого API или сервиса прокси к такому API. В `OPENAI_API_KEY` токен для доступа к этому API или прокси.

Использование:

```shell
python3 main.py "запрос в LLM"
```
