# Feature Specifications

Este directorio contiene las especificaciones de funcionalidades del proyecto Bakery.

## Qué es una Spec

Una spec documenta una funcionalidad implementada: su API, uso, configuración y ejemplos. A diferencia de los ADRs (que son inmutables), las specs evolucionan con la funcionalidad.

## Formato

Cada spec incluye:

- **Overview:** Descripción breve de la funcionalidad
- **API:** Clases, funciones y sus parámetros
- **Usage:** Ejemplos de código
- **CLI:** Argumentos de línea de comandos (si aplica)
- **Testing:** Cómo ejecutar tests relacionados
- **Related:** Links a ADRs, código fuente y tests

## Índice

| Spec | Descripción | Version |
|------|-------------|---------|
| [focus-lens.md](focus-lens.md) | Crop-based inference para concentrar píxeles | 1.0.0 |
| [model-catalog.md](model-catalog.md) | Sistema centralizado de gestión de modelos | 1.0.0 |

## Convenciones

- Nombres en kebab-case: `focus-lens.md`, `dual-model-pipeline.md`
- Incluir versión y fecha de última actualización
- Ejemplos de código ejecutables
- Referencias a tests para validar comportamiento
