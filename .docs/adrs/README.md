# Architecture Decision Records (ADRs)

Este directorio contiene los Architecture Decision Records del proyecto Bakery.

## Qué es un ADR

Un ADR documenta una decisión arquitectónica significativa junto con su contexto y consecuencias. Los ADRs son inmutables una vez aceptados; para cambiar una decisión, se crea un nuevo ADR que "supersede" al anterior.

## Formato

Cada ADR sigue el template en `000-template.md`:

- **Status:** Proposed → Accepted → [Deprecated | Superseded]
- **Context:** El problema que motivó la decisión
- **Decision:** Lo que decidimos hacer
- **Consequences:** Impacto positivo, negativo y neutral
- **Alternatives:** Otras opciones consideradas y por qué se rechazaron

## Índice

| ADR | Título | Status |
|-----|--------|--------|
| [000](000-template.md) | Template | N/A |
| [004](004-focus-lens-crop-based-inference.md) | Focus Lens - Crop-Based Inference | Accepted |
| [005](005-model-catalog-configuration.md) | Model Catalog Configuration System | Accepted |

## Convenciones

- Numerar secuencialmente: `001`, `002`, etc.
- Título descriptivo en kebab-case: `004-focus-lens-crop-based-inference.md`
- Escribir en tiempo presente para decisiones activas
- Incluir fecha y autores
