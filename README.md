# Explorando classificadores: Uma abordagem em Sistemas Inteligentes


Neste trabalho, foi proposto pelo professor Paulo Cirillo um problema de classificacao de expressoes faciais a partir de dados de 2 sensores localizados no Corrugador do Supercílio (Sensor 1); Zigomático Maior (Sensor 2).
No conjunto de dados disponibilizado pelo professor, existem 50 000 dados referentes a 5 expressões faciais forçadas (Neutro, Sorriso, Aberto, Surpreso, Grumpy).  

![[Pasted image 20230323221801.png]]



![[Pasted image 20230323190333.png]]
Ao observar o grafico, podemos tirar a conclusao que esse problema classificatorio eh possivel ser resolvido por meio de um modelo de classificacao, pois podemos observar que as classes podem ser separadas e sao determinadas por algum padrao que devemos descobrir.

**Acuracia**
| Classificador | Media | Menor Valor | Maior Valor| Desvio Padrao |
|---|---|---|---|---|
| OLS | 72.33 % | 70.99 % | 73.20 % | 0.64 |
| OLS Regularizado | 72.20% | 71.15% | 73.14% | 0.61 |
| Naive Bayes | 95.95% | 95.61% | 96.22% | 0.22 |
| Pooled | NULL | NULL | NULL | NULL |
| Friedman | 99.35 % | 99.26 % | 99.49 % | 0.08 |

**Tempo de Treino**
| Classificador | Media | Menor Valor | Maior Valor| Desvio Padrao |
|---|---|---|---|---|
| OLS | 3.57 ms | 2.98 ms | 4.13 ms | 0.49 |
| OLS Regularizado | 4.24 ms | 2.99 ms | 6.01 ms | 0.94 |
| Naive Bayes | 610.17 ms | 559.75 ms | 668.88 ms | 33.51 |
| Pooled | NULL | NULL | NULL | NULL |
| Friedman | 586.97 ms | 552.35 ms | 658.99 ms | 35.38 |

**Tempo de Execução**
| Classificador | Media | Menor Valor | Maior Valor| Desvio Padrao |
|---|---|---|---|---|
| OLS | 0.40 ms | 0.00 ms | 1.02 ms | 0.49 |
| OLS Regularizado | 0.30 ms | 0.00 ms | 1.02ms | 0.47 |
| Naive Bayes | 0.00 ms | 0.00 ms | 0.00 ms | 33.51 |
| Pooled | NULL | NULL | NULL | NULL |
| Friedman | 1.50 ms | 1.00 ms | 2.00 ms | 0.5 |





MMQ
- Usei a formula "Y = BX", sendo o Y a matriz de respostas com a magnitude (N, 5), X a matriz de dados com a magnitude (N, 3 (2 + 1)) e B sendo o parametro de medicao, que deve ser ajustado para obter uma maior acuracia, de magnitude (3, 5)
- Eu obtive uma taxa de acerto media de 72%
- O maior valor da taxa de acerto foi em media 74%
- A menor valor da taxa de acerto foi em media 70%
- Tempo de treino medio foi 2ms
- Tempo de previsao medio foi 0.15ms

pseudocodigo
```
separar os dados teste e treino
estimar o valor de B com os dados de treino (B = ((X.T@X)^-1)@X.T@Y)
computar o tempo de execucao do algoritmo de treino
obter os resultados de y para os dados de teste
verificar a quantidade de vezes que o modelo acertou
computar a acuracia
mostrar os resultados
```

MMQ Regularizado
- Eu utilizei a formula do MMQ Regularizado para estimar o parametro W (estimar o modelo) e depois testa-lo. (Y = WX), (W = )
- ![[Pasted image 20230321145228.png]] (Formula da predicao)
- ![[Pasted image 20230321145400.png]] (Formula do teste)
- Observei que os valores de acuracia sao bem parecidos com o MMQ ordinario e que a acuracia nao muda ao mudar o valor de lambda
- Eu obtive uma taxa de acerto media de 72%
- O maior valor da taxa de acerto foi em media 74%
- A menor valor da taxa de acerto foi em media 70%
- Tempo de treino medio foi 2ms
- Tempo de previsao medio foi 0.15ms
- 
pseudocodigo
```
separar os dados teste e treino
estimar W com os valores possiveis de lambda
verificar qual lambda da a maior acuracia
definir o lambda que da a maior acuracia
estimar o valor de W com os dados de treino e o lambda definido
computar o tempo de execucao do algoritmo de treino
obter os resultados de y para os dados de teste
verificar a quantidade de vezes que o modelo acertou
computar a acuracia
mostrar os resultados
```

Naves Bayes
[Explicacao](https://www.vooo.pro/insights/6-passos-faceis-para-aprender-o-algoritmo-naive-bayes-com-o-codigo-em-python/)

- Eh um modelo baseado na probabilidade de uma classe pertencer a dadas caracteristicas
- Ele fala que os preditores(sensor 1 e sensor2) sao independentes
- Por exemplo, para uma Laranja ser uma Laranja, ela precisa ter a cor laranja, ser redonda e ter mais ou menos 10cm de diametro. Esses fatores sao responsaveis por determinar que o objeto eh uma laranja mas contribuem de forma independente para determinar a classe
- 

![[Pasted image 20230322081031.png]]

- P(y|x) = P(x|y) * P(y) / P(x)
- P(y|x) = Probabilidade de ser a classe (c) dada as caracteristicas (x)
- P(y) = Probabilidade de uma condicao aleatoria pertencer a uma classe Y antes de ver a caracteristica
- P(x) nao entendi

a media representa o centro de distribuicao das classes
a variancia representa  a dispersao dos valores em relacao a media

a media e a variancia sao calculadas para cada caracteristica (sensor 1 e sensor 2)
a variancia eh usada para calcular a probabilidade de certa caracteristica pertencer a determinada classe

Primeiro Passo: Calcular a frequencia de cada classe

![[Pasted image 20230322194859.png]]
A diagonal principal da matriz de covariancia contém as variâncias de cada variável, que medem a dispersão dos valores em torno da média (distancia dos valores para o centro dos dadoo).
O mean significa a media dos dados para cada subconjunto de dados (cada classe tem um subconjunto)

Cada funcao discriminante calcula a probabilidade do dado xn pertencer a classe i 
Eh feita a discriminante para cada classe, a de maior valor deve ser a classe definida

A regularizacao eh feita para nao deixar que a matriz de covariancia seja singular (nao eh possivel inverter)

