# LegalMove — Agente Autónomo de Comparación de Contratos

Sistema multi-agente que procesa imágenes escaneadas de contratos legales y sus enmiendas, extrae el texto mediante visión multimodal (GPT-4o) y produce un análisis estructurado y validado de todos los cambios introducidos.

---

## Arquitectura del sistema

```
Entrada: imagen contrato original + imagen enmienda
         │                          │
         ▼                          ▼
┌─────────────────────────────────────────────┐
│           image_parser.py                   │
│  GPT-4o Vision — base64 → texto estructurado│
│  • parse_original_contract (span Langfuse)  │
│  • parse_amendment_contract (span Langfuse) │
└────────────────┬────────────────────────────┘
                 │ texto original + texto enmienda
                 ▼
┌─────────────────────────────────────────────┐
│       Agente 1: ContextualizationAgent      │
│  Rol: Analista Senior de Contratos Legales  │
│  Salida: mapa contextual JSON               │
│  • document_type, parties, contract_date    │
│  • general_purpose, structure_summary       │
│  (span "contextualization_agent" Langfuse)  │
└────────────────┬────────────────────────────┘
                 │ mapa contextual + ambos textos
                 ▼
┌─────────────────────────────────────────────┐
│        Agente 2: ExtractionAgent            │
│  Rol: Auditor Legal — Análisis de Cambios   │
│  Salida: ContractChangeOutput (Pydantic)    │
│  • sections_changed, topics_touched         │
│  • summary_of_the_change                   │
│  (span "extraction_agent" Langfuse)         │
└────────────────┬────────────────────────────┘
                 │
                 ▼
        JSON validado por Pydantic v2

Traza Langfuse completa:
contract-analysis (root trace)
├── parse_original_contract
├── parse_amendment_contract
├── contextualization_agent
└── extraction_agent
```

---

## Diagrama de módulos

```
src/
│
├── main.py
│   ├── Responsabilidad : orquestación del pipeline completo
│   ├── Entradas        : rutas de imagen (CLI argv)
│   ├── Salidas         : ContractChangeOutput → stdout JSON
│   ├── Clientes init   : OpenAI(), Langfuse() (una sola instancia)
│   ├── Traza raíz      : langfuse.trace("contract-analysis")
│   └── Dependencias    : image_parser, agents, models
│
├── image_parser.py
│   ├── Responsabilidad : extracción de texto de imagen via GPT-4o Vision
│   ├── Función pública : parse_contract_image(image_path, ...)
│   ├── Mecanismo       : imagen → base64 → API OpenAI (detail="high")
│   ├── Prompt system   : PARSING_SYSTEM_PROMPT (10 reglas estrictas)
│   ├── Retry logic     : exponential backoff (RateLimitError, Timeout)
│   ├── Span Langfuse   : text_length, tokens, latency_ms
│   └── Formatos        : jpg, jpeg, png, gif, webp
│
├── models.py
│   ├── Responsabilidad : schema de salida validado
│   ├── Clase           : ContractChangeOutput (Pydantic BaseModel)
│   ├── Campos          : sections_changed, topics_touched, summary_of_the_change
│   └── Validadores     : non_empty_list(), summary_min_length(≥50 chars)
│
└── agents/
    │
    ├── __init__.py
    │   └── Exports : ContextualizationAgent, ExtractionAgent
    │
    ├── contextualization_agent.py
    │   ├── Responsabilidad : construir mapa estructural del documento
    │   ├── Clase           : ContextualizationAgent
    │   ├── Método          : run(original_text, amendment_text, parent_trace)
    │   ├── LLM             : ChatOpenAI(gpt-4o, temperature=0)
    │   ├── Salida          : dict JSON con 5 campos estructurales
    │   ├── Fallback        : contexto mínimo si JSON inválido
    │   └── Span Langfuse   : document_type, parties_count, sections_mapped
    │
    └── extraction_agent.py
        ├── Responsabilidad : identificar y clasificar cada cambio
        ├── Clase           : ExtractionAgent
        ├── Método          : run(original_text, amendment_text, context_map, parent_trace)
        ├── LLM             : ChatOpenAI(gpt-4o, temperature=0)
        ├── Estrategia      : with_structured_output(ContractChangeOutput) + model_validate()
        ├── Salida          : ContractChangeOutput validado por Pydantic
        └── Span Langfuse   : sections_count, topics_count, latency_ms

Relaciones entre módulos:

  main.py
    ──calls──► image_parser.parse_contract_image()  ×2
    ──calls──► ContextualizationAgent.run()
    ──calls──► ExtractionAgent.run()
    ──uses ──► ContractChangeOutput (type hint)

  ContextualizationAgent
    ──uses──► langchain_openai.ChatOpenAI
    ──uses──► langchain_core.messages.{SystemMessage, HumanMessage}
    ──produces──► dict  ──consumed by──► ExtractionAgent

  ExtractionAgent
    ──uses──► langchain_openai.ChatOpenAI.with_structured_output()
    ──uses──► src.models.ContractChangeOutput
    ──uses──► pydantic.ValidationError
    ──produces──► ContractChangeOutput

Flujo de datos:

  [JPG/PNG] ──base64──► GPT-4o Vision ──str──► ContextAgent ──dict──► ExtractionAgent ──Pydantic──► JSON
```

---

## Fundamentos del mecanismo de tokenización

### ¿Qué es un token?

Un token es la unidad mínima de procesamiento que el LLM "lee" y "escribe". No equivale a una palabra ni a un carácter; es una secuencia de caracteres frecuentes en el corpus de entrenamiento. GPT-4o usa el tokenizador **cl100k_base** (tiktoken), que contiene ~100.000 tokens.

```
"contrato"      → 1 token
"confidencialidad" → 3 tokens  (confiden + cial + idad)
" Cláusula"     → 2 tokens  (espacio + Cl + áusula)
```

### Por qué importa en este sistema

Cada etapa del pipeline tiene un presupuesto de tokens que afecta directamente el costo y la calidad:

```
Etapa                    Entrada típica            Tokens aprox.
─────────────────────────────────────────────────────────────────
parse_original_contract  imagen 1.5MB (detail=high)  1.000–2.000 prompt
                                                       500–1.500 completion
parse_amendment_contract idem                         idem
contextualization_agent  2 textos (~1.500 chars c/u)  800–1.200 prompt
                                                       300–600 completion
extraction_agent         2 textos + context_map       2.000–4.000 prompt
                                                       500–1.500 completion
─────────────────────────────────────────────────────────────────
Total por ejecución                                  ~5.000–10.000 tokens
Costo estimado GPT-4o                                ~$0.05–$0.10 USD
```

### Cómo GPT-4o tokeniza imágenes (Vision)

Con `detail="high"`, GPT-4o divide la imagen en tiles de 512×512 px. El costo en tokens se calcula así:

```
1. La imagen se escala para que el lado más largo sea ≤ 2048 px.
2. Se divide en tiles de 512×512.
3. Cada tile = 170 tokens fijos + 85 tokens base.

Ejemplo: imagen 1024×1024
  → 4 tiles
  → 4 × 170 + 85 = 765 tokens de imagen
```

Con `detail="low"` el costo es siempre 85 tokens, pero se pierde resolución de cláusulas — inaceptable para texto legal denso.

### Límite de contexto y ventana efectiva

`gpt-4o` tiene una ventana de **128.000 tokens**. En este sistema el cuello de botella es el `extraction_agent`, que recibe:

```
context_window = len(system_prompt) + len(context_map_json)
               + len(original_text) + len(amendment_text)
               + len(respuesta_esperada)
```

Para contratos de más de 10 páginas el texto extraído puede superar los 6.000 tokens, lo que aún deja margen holgado. Si el documento supera las 30 páginas, ver la sección de escalado.

### Implicaciones para el costo

Langfuse registra `prompt_tokens` y `completion_tokens` por span, lo que permite calcular el costo exacto por ejecución y detectar llamadas anómalas (texto extraído muy largo, loops accidentales, prompts inflados).

---

## Optimización de prompts

### Principios aplicados en este sistema

**1. Separación de responsabilidades por agente**

Cada agente tiene un único trabajo declarado en la primera línea del system prompt. Mezclar contextualización y extracción en un solo agente aumenta la tasa de alucinaciones y degrada la exhaustividad.

```python
# ContextualizationAgent — CORRECTO
"TU RESPONSABILIDAD ÚNICA: construir un mapa contextual preciso."

# Incorrecto (anti-patrón): pedir contexto Y cambios en un solo llamado
```

**2. Role prompting con credenciales específicas**

Asignar un rol con experiencia concreta mejora la calidad del razonamiento legal:

```python
# Más efectivo
"Eres un Analista Senior de Contratos Legales con 15 años de experiencia
en derecho corporativo internacional."

# Menos efectivo
"Eres un asistente legal."
```

**3. Output format explícito con ejemplos inline**

Especificar el formato de salida con ejemplos dentro del prompt elimina ambigüedad y reduce el parsing post-respuesta:

```python
'"structure_summary": {"Cláusula 1": "presente en ambos",
                       "Cláusula 7": "nueva en enmienda"}'
```

**4. Restricciones negativas explícitas**

Decirle al LLM qué NO debe hacer es tan importante como decirle qué sí debe hacer:

```python
"- NO extraigas cambios de contenido específicos — eso lo hace el agente de extracción."
"- No inventes cambios — solo reporta lo que puedes verificar en el texto."
```

**5. Temperatura cero para tareas deterministas**

Ambos agentes y el parser usan `temperature=0`. En análisis legal la reproducibilidad importa más que la creatividad.

**6. Contexto descendente (context injection)**

El `extraction_agent` recibe el mapa del `contextualization_agent` como contexto explícito. Esto reduce el trabajo que el LLM debe hacer "de cero" y enfoca el razonamiento:

```python
human_content = f"""MAPA CONTEXTUAL (elaborado por el Analista Senior):
---
{context_str}
---
CONTRATO ORIGINAL:
...
"""
```

### Técnicas adicionales para mejorar resultados

| Técnica | Aplicación en este sistema | Ganancia esperada |
|---|---|---|
| Chain-of-thought implícito | "Sé exhaustivo: un cambio no detectado puede tener consecuencias legales graves" | Fuerza razonamiento paso a paso sin tokens extras |
| Anclaje en identificadores exactos | "Usa los identificadores exactos del mapa contextual" | Elimina secciones inventadas |
| Cuantificación obligatoria | "Para campos numéricos indica el valor anterior y el nuevo" | Previene resúmenes vagos |
| Priorización por impacto | "Prioriza: indemnizaciones, limitaciones de responsabilidad, plazos, honorarios" | Mejora el orden del summary |
| Validación downstream con Pydantic | `with_structured_output()` + `model_validate()` | Segunda capa de corrección de formato |

### Cómo medir la efectividad de un prompt

Usando Langfuse, se puede comparar variantes de prompt de forma sistemática:

```python
# Registrar metadata de versión del prompt en cada span
span.end(
    metadata={
        "prompt_version": "v2.1",
        "technique": "role+constraints+examples",
    }
)
```

Luego en el dashboard de Langfuse filtrar por `prompt_version` y comparar:
- `sections_count` promedio (exhaustividad)
- `latency_ms` promedio (eficiencia)
- `total_tokens` promedio (costo)

---

## Propuesta de escalado

### Escenario actual (baseline)

```
1 imagen par → pipeline secuencial → ~30-60 seg → ~$0.05–$0.10 USD
```

El sistema actual es correcto y funcional para validación y demos. Las siguientes propuestas escalan según el volumen de uso.

---

### Nivel 1 — Paralelización interna (0–100 pares/día)

**Problema**: las dos llamadas de parsing son secuenciales aunque son independientes.

**Solución**: ejecutar ambos parsings en paralelo con `asyncio` o `ThreadPoolExecutor`:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    fut_original  = executor.submit(parse_contract_image, original_path, ...)
    fut_amendment = executor.submit(parse_contract_image, amendment_path, ...)
    original_text  = fut_original.result()
    amendment_text = fut_amendment.result()
```

**Ganancia estimada**: reducción de latencia total del 30–40% sin cambiar infraestructura.

---

### Nivel 2 — API con cola de trabajos (100–1.000 pares/día)

**Arquitectura**:

```
Cliente HTTP
    │
    ▼
FastAPI (POST /analyze)
    │
    ▼
Cola de mensajes (Redis / RabbitMQ / SQS)
    │
    ├──► Worker 1: run_pipeline()
    ├──► Worker 2: run_pipeline()
    └──► Worker N: run_pipeline()
    │
    ▼
Base de datos de resultados (PostgreSQL)
    │
    ▼
GET /result/{job_id}  ←── polling del cliente
```

**Componentes nuevos**:

| Componente | Tecnología sugerida | Rol |
|---|---|---|
| API Gateway | FastAPI + Pydantic | Validación de entrada, autenticación |
| Cola | Redis Streams / AWS SQS | Desacoplamiento productor/consumidor |
| Workers | Celery / ARQ / AWS Lambda | Ejecución paralela del pipeline |
| Almacenamiento de imágenes | AWS S3 / GCS | Evitar transferencia base64 en memoria |
| Resultados | PostgreSQL + pgvector | Historial + búsqueda semántica futura |
| Observabilidad | Langfuse (ya integrado) | Sin cambios necesarios |

**Cambio clave en el pipeline**: reemplazar lectura de archivo local por descarga desde S3:

```python
# Actual
b64 = base64.b64encode(Path(image_path).read_bytes())

# Escalado
import boto3
s3 = boto3.client("s3")
obj = s3.get_object(Bucket=bucket, Key=key)
b64 = base64.b64encode(obj["Body"].read())
```

---

### Nivel 3 — Procesamiento de documentos multipágina (contratos largos)

**Problema**: contratos reales suelen tener 20–80 páginas. Una sola imagen no alcanza para preservar toda la resolución.

**Solución — Chunking por sección**:

```
PDF/TIFF multipágina
    │
    ▼
Splitter: divide en páginas individuales (pdf2image / pypdfium2)
    │
    ├──► página 1 → parse_contract_image()
    ├──► página 2 → parse_contract_image()
    └──► página N → parse_contract_image()
    │
    ▼
Merger: concatena textos preservando saltos de página
    │
    ▼
Pipeline actual (contextualization + extraction)
```

**Consideración de tokens**: con 80 páginas, el texto concatenado puede superar 40.000 tokens. Estrategias:
- Usar `gpt-4o` (128k context) — sin cambios, funciona hasta ~60 págs.
- Para documentos mayores: chunking semántico por cláusula + map-reduce sobre extraction_agent.

---

### Nivel 4 — Búsqueda y auditoría histórica (1.000+ pares/día)

**Problema**: en una firma legal se procesan cientos de contratos. Necesitan buscar "todos los contratos donde cambió la cláusula de confidencialidad".

**Solución — Vector store + RAG**:

```
ContractChangeOutput
    │
    ├── summary_of_the_change ──► embed (text-embedding-3-small)
    │                              └──► pgvector / Pinecone
    │
    └── sections_changed, topics_touched ──► índice filtrable
                                             (Elasticsearch / PostgreSQL FTS)

Consulta:
  "contratos con cambios en honorarios mayores a $5.000"
      │
      ▼
  Búsqueda híbrida (vector + filtro estructurado)
      │
      ▼
  Top-K resultados con score de relevancia
```

---

### Nivel 5 — Multi-modelo y reducción de costos

Para reducir costos en producción sin sacrificar calidad crítica:

```
Estrategia de enrutamiento por complejidad:

image_parser
  ├── doc simple (1 pág, texto claro) ──► gpt-4o-mini  ($0.15/1M tokens)
  └── doc complejo (multipág, tablas) ──► gpt-4o       ($2.50/1M tokens)

contextualization_agent ──► gpt-4o-mini  (tarea estructurada, baja ambigüedad)
extraction_agent        ──► gpt-4o       (tarea crítica, máxima precisión)
```

**Ahorro estimado**: 40–60% de reducción de costo por ejecución con calidad equivalente en el 80% de los documentos.

---

### Resumen de la hoja de ruta de escalado

```
Volumen          Nivel  Cambio principal                  Esfuerzo
────────────────────────────────────────────────────────────────────
< 100 pares/día    1    Paralelizar los 2 parsers          1 día
100–1K pares/día   2    FastAPI + cola + workers           1 semana
1K–10K pares/día   3    Soporte PDF multipágina + chunking 1 semana
> 10K pares/día    4    Vector store + búsqueda histórica  2 semanas
Optimización       5    Enrutamiento multi-modelo          3 días
```

---

## Estructura del proyecto

```
aem4/
├── src/
│   ├── main.py                          # Entry point del pipeline
│   ├── image_parser.py                  # Parsing multimodal GPT-4o Vision
│   ├── models.py                        # Schema Pydantic ContractChangeOutput
│   └── agents/
│       ├── __init__.py
│       ├── contextualization_agent.py   # Agente 1: contexto
│       └── extraction_agent.py          # Agente 2: extracción de cambios
├── data/
│   └── test_contracts/                  # 3 pares de imágenes de prueba
├── .env.example                         # Template de variables de entorno
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
```

Editar `.env` con las claves reales:

```env
OPENAI_API_KEY=sk-...          # API key de OpenAI
LANGFUSE_PUBLIC_KEY=pk-lf-...  # Clave pública de Langfuse
LANGFUSE_SECRET_KEY=sk-lf-...  # Clave secreta de Langfuse
LANGFUSE_HOST=https://cloud.langfuse.com
```

**Obtener claves Langfuse:**
1. Crear cuenta en https://cloud.langfuse.com
2. Crear un nuevo proyecto
3. En Settings → API Keys, copiar public key y secret key

---

## Uso

```bash
python src/main.py <imagen_original> <imagen_enmienda>
```

### Ejemplos con los contratos de prueba

**Par 1 — Cambio simple (Contrato de Licencia de Software):**
```bash
python src/main.py \
  data/test_contracts/documento_1__original.jpg \
  data/test_contracts/documento_1__enmienda.jpg
```
Cambios esperados: plazo 12→24 meses, tarifa anual, soporte ampliado, cláusula de protección de datos nueva.

**Par 2 — Cambios múltiples (Contrato de Servicios de Consultoría):**
```bash
python src/main.py \
  data/test_contracts/documento_2__original.jpg \
  data/test_contracts/documento_2__enmienda.jpg
```
Cambios esperados: duración 6→9 meses, honorarios $8.000→$9.500, entregables quincenales, propiedad intelectual nueva.

**Par 3 — Contrato SaaS:**
```bash
python src/main.py \
  data/test_contracts/documento_3__original.jpg \
  data/test_contracts/documento_3__enmienda.jpg
```
Cambios esperados: precio $1.200→$1.250, disponibilidad 99,5%→99,9%, soporte ampliado con sistema de tickets.

---

## Salida del sistema

```json
{
  "sections_changed": [
    "Cláusula 2 — Plazo",
    "Cláusula 3 — Pago",
    "Cláusula 4 — Soporte",
    "Cláusula 7 — Protección de Datos"
  ],
  "topics_touched": [
    "Plazo del contrato",
    "Honorarios y tarifas",
    "Soporte técnico",
    "Protección de datos personales"
  ],
  "summary_of_the_change": "La enmienda introduce cuatro modificaciones sobre el contrato original..."
}
```

---

## Decisiones técnicas

### Por qué GPT-4o Vision
GPT-4o es el único modelo de OpenAI con soporte de visión de alta fidelidad para documentos densos en texto. Con `detail: "high"`, preserva numeración de cláusulas, términos definidos y estructura jerárquica — elementos críticos para el análisis legal. Se usa base64 en lugar de URLs para portabilidad y funcionamiento offline.

### Por qué 2 agentes separados
Un solo agente que contextualice y extraiga cambios al mismo tiempo degrada la calidad de ambas tareas. La separación de responsabilidades permite:
- **Agente 1 (Analista)**: enfocarse en entender qué ES el documento, sin comparar.
- **Agente 2 (Auditor)**: recibir un mapa ya construido y enfocarse exclusivamente en QUÉ cambió.

Este patrón reduce alucinaciones y mejora la exhaustividad de la extracción.

### Por qué Pydantic v2 con with_structured_output()
`with_structured_output()` de LangChain pasa el schema Pydantic como definición de función al LLM, forzando una respuesta JSON conforme al schema antes de la deserialización. La validación explícita adicional con `model_validate()` agrega una segunda capa de seguridad. Los `field_validator` personalizan los mensajes de error para el dominio legal.

### Por qué Langfuse
Langfuse permite trazar la ejecución completa con jerarquía padre-hijo (trace → spans), capturando inputs, outputs, latencias y tokens por etapa. Esto es esencial en producción para:
- Debuggear qué etapa falló en una ejecución específica
- Auditar qué texto extrajo el parser y qué vio cada agente
- Monitorear costos por imagen procesada

---

## Observabilidad en Langfuse

Cada ejecución crea una traza en el dashboard de Langfuse con esta jerarquía:

```
contract-analysis
├── parse_original_contract    → texto extraído, tokens de imagen, latencia
├── parse_amendment_contract   → idem
├── contextualization_agent    → mapa contextual, tokens LLM, latencia
└── extraction_agent           → secciones detectadas, output Pydantic, latencia
```

**Métricas disponibles por span:**
- `latency_ms`: tiempo de respuesta de cada llamada
- `prompt_tokens` / `completion_tokens`: costo de cada etapa
- `text_length`: longitud del texto extraído por el parser
- `sections_count`: número de cambios detectados por el auditor
