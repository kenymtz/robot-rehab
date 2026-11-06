# robot-rehab

¿Puede un robot simulado volver a caminar después de perder parte de sus neuronas, usando solo un
registro de su comportamiento sano? Todo ocurre en simulación: no hay robots ni chips físicos.

## Correr

Requisitos: Linux x86-64, `bash`, `curl`, 5 GB libres e internet la primera vez.

```bash
./run.sh              # corrida completa, se separa de la terminal
./run.sh --foreground # igual, sin separarse
./run.sh --smoke      # recorrido completo en miniatura, sin validez científica
```

El progreso queda en `run.log`. Si la corrida se corta, volvé a ejecutar `./run.sh` y sigue desde
donde quedó. Al terminar, `results.zip` contiene el informe, los gráficos, los datos y el video.
