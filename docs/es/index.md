# agenteval en español

**Ejecuta tu agente LLM muchas veces, registra cada paso y descubre por qué falla.**

Un agente que responde bien en una demo puede ser lento, caro o inestable, o acertar por la
razón equivocada: respondió de memoria en vez de usar sus herramientas, o un límite de tasa
hizo fallar la mitad de sus ejecuciones. Un solo número de precisión esconde todo eso.

agenteval ejecuta tu agente sobre un conjunto de preguntas, varias veces cada una, registra
cada llamada al modelo y cada llamada a herramientas, y te dice:

- **cuántas veces acierta**, en total y por tipo de pregunta;
- **por qué falla**: respuesta incorrecta, error de una herramienta, timeout, límite de tasa,
  cuota agotada, una instrucción inyectada que obedeció. Cada fallo lleva su causa;
- **cómo usa sus herramientas**: la correcta, con argumentos válidos, en el orden correcto,
  sin pasos de más, y si su respuesta se apoya en lo que le devolvieron;
- **cuánto cuesta y cuánto tarda**, por respuesta correcta y en p50/p95;
- **si un cambio lo mejoró de verdad**, con una prueba estadística pareada, para no publicar
  un ajuste de prompt que solo pareció mejor por suerte.

```bash
pip install "pyagenteval[server]"
agenteval init      # un agente de ejemplo, unas preguntas y una configuración
agenteval run       # el ejemplo no necesita llave de API
agenteval ui        # míralo en el navegador
```

## Para qué sirve

No es para entrenar modelos: es para probar **aplicaciones hechas con LLMs** (agentes con
herramientas, chatbots, sistemas RAG) igual que pruebas el resto de tu código. Funciona con
cualquier agente: una función tuya, el SDK de OpenAI o de Anthropic, LangChain y LangGraph,
o cualquier framework con OpenTelemetry (LlamaIndex, OpenAI Agents SDK, CrewAI, PydanticAI...).

## Por dónde seguir

- **[Inicio rápido](quickstart.md)**: tu propio agente evaluado en cinco minutos.
- **[Tutorial](tutorial.md)**: un agente de soporte, desde la primera ejecución hasta producción.
- El resto de la documentación está en inglés: [conectar tu agente](../guides/your-agent.md),
  [métricas](../guides/metrics.md), [comparar versiones](../guides/compare.md),
  [producción](../guides/production.md).

!!! note "Nombres"
    El proyecto se llama agent-eval, el paquete de Python y el comando son `agenteval`, y se
    instala desde PyPI como `pyagenteval`.
