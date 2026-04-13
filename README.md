# Calisthenics AI Trainer (Tesis de Maestría)

Este repositorio contiene la implementación actual y el avance del dataset para nuestro proyecto de tesis de maestría sobre análisis de movimientos de calistenia con MediaPipe.

Autores: **José Guambo** y **Aaron Echeverría**

## Arquitectura

### Versión 1

![Arquitectura versión 1](docs/architecture/arquitectura-v1.png)

### Versión 2

![Arquitectura versión 2](docs/architecture/arquitectura-v2.png)

### Versión 3

![Arquitectura versión 3](docs/architecture/arquitectura-v3.png)

### Versión 4

![Arquitectura versión 4](docs/architecture/arquitectura-v4.png)

## Bitácora de Decisiones Arquitectónicas

### Versión 1: prueba local de factibilidad

En la primera versión trabajamos de forma completamente local. El objetivo era validar rápidamente si MediaPipe, la primera librería que encontramos para estimación de pose, podía detectar de manera confiable los landmarks o articulaciones del cuerpo humano en videos de calistenia.

Esta etapa fue una prueba de factibilidad. Antes de diseñar una arquitectura distribuida, necesitábamos confirmar que la base técnica del proyecto era viable y que la detección de pose funcionaba para nuestros movimientos objetivo.

### Versión 2: diseño inicial con WhatsApp y Azure

En la segunda versión diseñamos un flujo cuyo punto de entrada era WhatsApp. La mayor parte de los servicios estaba pensada para ejecutarse en Azure:

- Azure Blob Storage para almacenar videos sin procesar y videos procesados.
- Azure Functions para ejecutar la lógica de procesamiento.
- Azure Cognitive Services como base para la etapa de pose estimation.
- Azure Service Bus para orquestar el envío de la respuesta final.

La idea era que el video ingresara por WhatsApp, se almacenara en Blob Storage, se activara el procesamiento mediante triggers y, al finalizar, se enviara una respuesta al usuario con el video analizado y correcciones técnicas en texto. Esta propuesta aprovechaba especialmente la integración por eventos de Blob Storage.

### Versión 3: cambio de canal, simplificación operativa y reducción de costos

En la tercera versión hicimos varios ajustes importantes. El principal cambio fue abandonar WhatsApp Business como canal principal. Aunque logramos probar el envío de mensajes con la API de Meta, la configuración completa resultó demasiado compleja para nuestro contexto y la integración no nos permitió avanzar de forma fluida con videos.

Además, para pasar a un entorno productivo con WhatsApp Business era necesario presentar evidencia legal de la empresa y esperar procesos de revisión burocráticos. Eso bloqueaba el avance del proyecto.

Por esa razón migramos a Telegram. La configuración fue casi inmediata usando BotFather, donde definimos el nombre del bot, una descripción básica y obtuvimos una API key para pruebas del webhook. En ese punto, enviamos un video a `@CalistenIA_Entrenador_Bot` y el flujo funcionó correctamente desde el inicio.

También alquilamos un despliegue de n8n en Hostinger, con un costo aproximado de 16 dólares mensuales, y movimos allí la automatización principal. En esta etapa, todo el flujo vivía en la nube, excepto el analizador de video, que seguía ejecutándose en un computador personal. Además, reemplazamos Azure Blob Storage por Cloudflare para almacenar videos, principalmente por su free tier generoso y su salida de datos sin costo adicional.

### Versión 4: contenedorización completa y despliegue conjunto en VPS

La cuarta versión fue la consolidación operativa del sistema. Contenerizamos el analizador de video en Python y lo desplegamos en el mismo VPS de Hostinger donde ya estaba corriendo n8n, usando Docker Compose. El servidor utilizado corresponde al plan KVM 4, con 4 vCPU, 16 GB de memoria y 200 GB de espacio en disco.

Este cambio respondió a un problema de rendimiento muy claro. El flujo completo tardaba alrededor de 45 segundos y, al revisar los tiempos de transacción en n8n, vimos que más del 90% del tiempo estaba asociado al procesamiento remoto del video en la máquina local. Ese retraso provenía de tres factores:

- La transferencia del video por internet entre el VPS y la máquina local.
- El procesamiento en un equipo no dedicado, con otras tareas ejecutándose en background (Sistema Operativo Windows).
- La latencia al devolver el resultado otra vez al servidor.

Además, la máquina local debía permanecer encendida permanentemente y sufría fallos de red frecuentes debido a una conexión inestable. Al mover el analizador al mismo VPS, ambos servicios pudieron comunicarse por red interna. Como resultado, el tiempo total bajó aproximadamente a 7 segundos.

En esta versión también incorporamos optimizaciones de despliegue. Creamos un repositorio privado en GitHub y lo integramos con GitHub Actions para construir y publicar automáticamente una imagen en Docker Hub cuando hay cambios en el código. El proceso todavía no es completamente automático porque, después de publicar la imagen, aún debemos entrar al VPS y volver a ejecutar `docker compose` para que se haga un pull de la nueva imagen del contenedor de python para procesamiento de video.

Finalmente, definimos endpoints específicos para separar responsabilidades:

- `video/process`: endpoint llamado por n8n para analizar videos.
- `landmarks/generate`: endpoint interno para generar archivos CSV a partir de videos de ejemplo de ejecución perfecta del movimiento.
- `movement-model/train`: endpoint que toma los CSV, entrena el clasificador y genera un nuevo archivo `pose_landmarker_lite.task`.

Ese archivo `pose_landmarker_lite.task` es el modelo entrenado que usamos para la clasificación final de movimientos.

## Avance del Dataset (Movimiento x Ángulo)

Leyenda de estado:
- `✅`: existe al menos un archivo `.mp4` en esa carpeta de movimiento/ángulo.
- `❌`: aún no está completado.

| Movimiento | Espalda | Diagonal | Lado |
|---|---|---|---|
| double-swing-360 | ❌ | ❌ | ❌ |
| dragon-360 | ❌ | ✅ | ✅ |
| geinger | ✅ | ✅ | ✅ |
| handstand | ❌ | ✅ | ✅ |
| olympic-muscle-up | ✅ | ✅ | ✅ |
| pasavallas | ❌ | ❌ | ❌ |
| strict-muscle-up | ✅ | ✅ | ✅ |
| swing-360 | ✅ | ✅ | ✅ |
| torero | ❌ | ✅ | ❌ |

## Avance del Dataset de Evaluación (Movimiento x Ángulo)

Leyenda de estado:
- `✅`: existe al menos un archivo `.mp4` en esa carpeta de movimiento/ángulo.
- `❌`: aún no está completado.

| Movimiento | Espalda | Diagonal | Lado |
|---|---|---|---|
| double-swing-360 | ❌ | ✅ | ❌ |
| dragon-360 | ❌ | ❌ | ✅ |
| geinger | ❌ | ✅ | ❌ |
| handstand | ❌ | ❌ | ❌ |
| olympic-muscle-up | ❌ | ✅ | ❌ |
| pasavallas | ❌ | ❌ | ❌ |
| strict-muscle-up | ❌ | ❌ | ❌ |
| swing-360 | ❌ | ✅ | ❌ |
| torero | ✅ | ✅ | ❌ |

## Comparación Final de Evaluación (Movimiento x Ángulo)

Leyenda de estado:
- `✅`: tanto `movements` como `movements-evaluation` contienen al menos un `.mp4` para ese movimiento/ángulo.
- `❌`: aún no está disponible en ambos datasets.

| Movimiento | Espalda | Diagonal | Lado |
|---|---|---|---|
| double-swing-360 | ❌ | ❌ | ❌ |
| dragon-360 | ❌ | ❌ | ✅ |
| geinger | ❌ | ✅ | ❌ |
| handstand | ❌ | ❌ | ❌ |
| olympic-muscle-up | ❌ | ✅ | ❌ |
| pasavallas | ❌ | ❌ | ❌ |
| strict-muscle-up | ❌ | ❌ | ❌ |
| swing-360 | ❌ | ✅ | ❌ |
| torero | ❌ | ✅ | ❌ |
