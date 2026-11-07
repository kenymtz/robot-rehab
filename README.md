# robot-rehab

¿Puede un robot simulado volver a caminar después de perder parte de sus neuronas, usando un
registro de su comportamiento sano? Todo ocurre en simulación: no hay robots ni chips físicos.

Un Walker2d entrenado con SAC pierde neuronas ocultas de forma permanente. Después se ajustan los
pesos sobrevivientes contra pares observación-acción guardados cuando estaba sano. Durante la
reparación no hay recompensa, simulador ni gradientes, y el código sabe qué neuronas murieron.
La recuperación se mide haciéndolo caminar.

## Resultado

Corrida oficial: 3 robots, 20% de neuronas muertas, 5 patrones de daño reservados y 20 intentos por
caso. Recuperación es la parte del rendimiento perdido que volvió: 0% es quedar igual que el robot
dañado y 100% es volver al rendimiento sano. Veredicto de la regla fijada antes de correr: seguir.

| Condición | Recuperación, mediana | IC 95% | Intentos sin caerse |
|---|---|---|---|
| Sin reparar | no aplica | no aplica | 3/300 |
| Reescalado de actividad | 3% | -10% a 33% | 28/300 |
| Mínimos cuadrados, solo la salida | 35% | 14% a 56% | 95/300 |
| **Mutación con memoria (principal)** | **84%** | **68% a 98%** | **212/300** |
| Clonación con gradientes (referencia) | 99% | 97% a 101% | 293/300 |
| Mutación, memoria de 800 pares | 48% | 25% a 79% | 137/300 |
| Clonación, memoria de 800 pares | 80% | 19% a 93% | 179/300 |

El robot sano no se cayó en 59 de 60 intentos.

### Un error bajo no garantiza que camine

El error contra la memoria sirve para comparar métodos (correlación de Spearman -0,78 en 60 casos),
pero predice mucho menos dentro de la mutación (-0,41 en 15 casos). Tampoco hay un umbral común:
con error 0,025 el robot 0 no se cayó en 18 de 20 intentos, y con 0,0076 el robot 2 lo logró en 2.
Son pocos casos, así que es una observación y no una ley.

## Límites

- El robot reparado se cae en 29% de los intentos.
- La clonación con gradientes rinde más con la misma memoria, y ganó en 12 de 15 casos. La mutación
  usa unas 53 veces más pasadas. Este proyecto mide hasta dónde llega la reparación sin
  retropropagación; no propone un método mejor.
- La comparación no es pareja: la clonación elige sus pesos con pares reservados y la mutación no,
  y sus presupuestos son distintos.
- Vale para 20% de daño, una tarea, una arquitectura y 3 políticas.
- La falla es una máscara por software, al azar y con ubicación conocida.
- La memoria de 8.000 pares ocupa unas 2,6 veces lo que una copia de los pesos.
- El tamaño de los pasos y el presupuesto de la mutación se calibraron mirando caminata simulada en
  un robot de demostración, distinto de los tres oficiales.
- Los valores elegidos de la mutación quedaron en el borde de su grilla.

## Trabajos relacionados

No encontré un trabajo que combine todas estas condiciones en una política de control, pero casi
todas las piezas ya existen. La búsqueda no fue exhaustiva, así que esto no prueba que sea nuevo.
Los más cercanos:

- [Kim, Choi y Lee 2026](https://arxiv.org/abs/2609.19579): recupera políticas robóticas podadas de
  forma offline, sin recompensa ni simulador, pero con gradientes. Es un preprint.
- [GLUESTICK](https://arxiv.org/abs/2510.08464): recupera modelos podados sin entrenar, con una
  corrección de bajo rango que usa los pesos originales.
- [ContrRep](https://lyudeyun.github.io/pubs/GECCO2024.pdf): repara controladores neuronales con
  evolución diferencial, usando simulación y una especificación.
- [Liu et al.](https://eprints.whiterose.ac.uk/id/eprint/133817/1/TNNLS.pdf): redes de impulsos que
  se reparan con plasticidad local hacia una frecuencia prefijada.
- [Wan et al. 2021](https://arxiv.org/pdf/2111.04957): fallas permanentes en políticas de RL, con
  recuperación por recompensa.

Dos antecedentes evolutivos no se pudieron leer completos:
[Higuchi et al. 1996](https://link.springer.com/chapter/10.1007/3-540-61093-6_6) y
[Sugawara et al. 2004](https://global.ieice.org/en_transactions/information/10.1587/e87-d_8_2021/_p).

En el código, `asr` es el reescalado de actividad. No es Adaptive Signal Resuscitation.

## Correr

Requisitos: Linux x86-64, `bash`, `curl`, 5 GB libres e internet la primera vez.

```bash
./run.sh              # corrida completa, se separa de la terminal
./run.sh --foreground # igual, sin separarse
./run.sh --smoke      # recorrido completo en miniatura, sin validez científica
```

El progreso queda en `run.log`. Si la corrida se corta, volvé a ejecutar `./run.sh` y sigue desde
donde quedó. Al terminar, `results.zip` contiene el informe, los gráficos, los datos y el video.

La corrida oficial se hizo en una notebook con Windows y 8 procesos, y tardó unas 6 horas y media:

```bash
uv run --frozen python -m robotrehab.cli --config configs/full.yaml --workers 8
```

El análisis del error se reproduce después de esa corrida:

```bash
uv run --frozen python scripts/offline_error.py
```

## Licencia

MIT. Ver [LICENSE](LICENSE).
