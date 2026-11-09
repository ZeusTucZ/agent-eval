# Inicio rápido

## 1. Instala

```bash
pip install "pyagenteval[server]"
```

Python 3.11 o más nuevo. `[server]` añade el dashboard; sin él tienes la línea de comandos y
la API de Python. Otros extras: `[anthropic]` para Claude, `[otel]` para frameworks con
OpenTelemetry, `[gemini]`, `[rag]`, o `[all]` para todo.

## 2. Crea un proyecto

```bash
mkdir mi-eval && cd mi-eval
agenteval init
```

Esto escribe tres archivos:

| Archivo | Qué es |
|---|---|
| `my_agent.py` | un agente de ejemplo: una función que recibe una pregunta y devuelve una respuesta |
| `cases.jsonl` | cuatro preguntas con su respuesta esperada |
| `agenteval.toml` | qué agente, modelo, precio y dataset usa `agenteval run` |

Para empezar desde un agente más parecido al tuyo, elige una plantilla:
`agenteval init --template openai` (o `anthropic`, `langgraph`, `rag`). Cada una escribe un
agente que funciona y te dice qué instalar y qué llave configurar.

## 3. Ejecútalo

```bash
agenteval run
agenteval report
```

El ejemplo no necesita llave de API: su "modelo" son unas cuantas reglas. El reporte muestra
la tasa de aciertos, cada verificación, la latencia, el costo y los fallos.

## 4. Pon tu propio agente

Reemplaza el cuerpo de `answer` en `my_agent.py` por tu agente. Envuelve las llamadas al
modelo en `llm_call` y las de herramientas en `tool`, para que cada una sea un paso que
agenteval pueda medir, costear y calificar:

```python
from openai import AsyncOpenAI
from agenteval import llm_call, tool

client = AsyncOpenAI()

async def answer(question: str) -> str:
    with llm_call("gpt-4o-mini") as call:
        response = await client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": question}])
        call.tokens(response.usage.prompt_tokens, response.usage.completion_tokens)
    return response.choices[0].message.content
```

¿Usas un cliente de OpenAI o de Anthropic? `wrap_openai(client)` o `wrap_anthropic(client)`
registran cada llamada por ti; LangChain y LangGraph tienen un callback, y cualquier
framework con OpenTelemetry se conecta con `instrument()`. Ver
[Connect your agent](../guides/your-agent.md).

Luego pon el modelo y su precio en `agenteval.toml`:

```toml
[run]
agent = "my_agent:answer"
model = "gpt-4o-mini"
price = [0.15, 0.60]   # dólares por millón de tokens de entrada y de salida
```

`agenteval prices --update` descarga precios actuales de miles de modelos; después,
normalmente puedes omitir `price`.

## 5. Escribe tus preguntas

Un objeto JSON por línea en `cases.jsonl`. Solo `input` y `expected_output` son obligatorios:

```json
{"input": "¿Cuál es el plazo de devolución?", "expected_output": "30 días", "category": "políticas"}
```

¿Aún no tienes preguntas? `agenteval generate docs/` las escribe a partir de tus documentos.
Ver [Write a dataset](../guides/datasets.md).

## 6. Mira, cambia, compara

```bash
agenteval ui                 # explora ejecuciones, abre fallos, compara dos ejecuciones
agenteval compare 1 2        # ¿la ejecución 2 es de verdad mejor que la 1?
```

O desde Python o un notebook:

```python
from agenteval import evaluate
results = evaluate(answer, "cases.jsonl", model="gpt-4o-mini", price=(0.15, 0.60))
results            # el reporte (una tabla HTML en Jupyter)
results.failures() # los intentos que no pasaron, con pregunta y respuesta
results.check(min_pass_rate=0.9)  # en un test de pytest: falla si no se cumple
```

Siguiente paso: el [tutorial](tutorial.md), con un agente real de principio a fin.
