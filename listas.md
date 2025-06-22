
### **Lista de Exercícios 1**

**Questão 1: Pode haver um grafo simples com 15 vértices, cada um com grau 5?**

**Resposta:** Não.

**Justificativa:**

1.  [cite\_start]A **Propriedade 1 (Lema do Aperto de Mão)** afirma que a soma dos graus de todos os vértices em um grafo é igual a duas vezes o número de arestas ($\\sum\_{v \\in V} d(v) = 2m$). [cite: 547]
2.  Neste caso, teríamos 15 vértices, cada um com grau 5. A soma dos graus seria $15 \\times 5 = 75$.
3.  O número 75 é ímpar. No entanto, a soma dos graus deve ser igual a $2m$, que é um número obrigatoriamente par.
4.  [cite\_start]Alternativamente, a **Propriedade 2 (Corolário)** afirma que o número de vértices de grau ímpar em um grafo deve ser par. [cite: 583]
5.  Neste caso, teríamos 15 vértices de grau ímpar (grau 5), e 15 é um número ímpar, o que viola o corolário.
6.  Portanto, é impossível construir tal grafo.

-----

**Questão 2: Mostre que não existe grafo simples com 4 vértices de maneira que três desses vértices possuem grau 3 e um vértice possui grau 1.**

**Resposta:** Não existe tal grafo.

**Justificativa (Prova por Contradição):**

1.  Seja um grafo simples $G$ com $V = {v\_1, v\_2, v\_3, v\_4}$.
2.  Vamos assumir que $d(v\_1) = 3$, $d(v\_2) = 3$, $d(v\_3) = 3$ e $d(v\_4) = 1$.
3.  [cite\_start]Em um grafo simples, o grau máximo que um vértice pode ter é $n-1$. [cite: 646] Aqui, com $n=4$, o grau máximo é 3.
4.  Um vértice com grau 3 (como $v\_1$) deve ser adjacente a todos os outros 3 vértices. Portanto, $v\_1$ deve ser adjacente a $v\_2, v\_3$ e $v\_4$.
5.  Da mesma forma, o vértice $v\_2$ (com grau 3) deve ser adjacente a $v\_1, v\_3$ e $v\_4$.
6.  E o vértice $v\_3$ (com grau 3) deve ser adjacente a $v\_1, v\_2$ e $v\_4$.
7.  A partir dessas premissas, o vértice $v\_4$ é adjacente a $v\_1$, $v\_2$ e $v\_3$.
8.  Isso implica que o grau de $v\_4$ deve ser 3.
9.  Isso contradiz a condição inicial de que $d(v\_4) = 1$.
10. Logo, a suposição inicial é falsa e tal grafo não pode existir.

-----

**Questão 3: Mostre que em uma festa com n ($n \\ge 2$) pessoas, existem pelo menos duas pessoas com o mesmo número de conhecidos.**

**Resposta:** Sim, existem.

**Justificativa (Prova pelo Princípio da Casa dos Pombos):**

1.  Modelamos o problema como um grafo simples $G$, onde as $n$ pessoas são os vértices ($V$) e um aperto de mão (conhecido) é uma aresta ($E$). O "número de conhecidos" de uma pessoa é o grau do vértice correspondente.
2.  [cite\_start]Em um grafo simples com $n$ vértices, o grau de qualquer vértice $v$ pode variar de $0$ a $n-1$. [cite: 646]
3.  Temos $n$ pessoas (vértices), que são os "pombos". Os possíveis graus são as "casas".
4.  Analisamos dois casos para os possíveis valores de grau:
      * **Caso A:** Existe um vértice com grau $n-1$. Este vértice está conectado a todos os outros $n-1$ vértices. Portanto, não pode haver um vértice com grau 0 (um vértice isolado). Nesse caso, os possíveis graus para os $n$ vértices pertencem ao conjunto ${1, 2, ..., n-1}$. Temos $n$ vértices e $n-1$ possíveis graus. Pelo Princípio da Casa dos Pombos, pelo menos dois vértices devem ter o mesmo grau.
      * **Caso B:** Não existe um vértice com grau $n-1$. Nesse caso, os possíveis graus para os $n$ vértices pertencem ao conjunto ${0, 1, ..., n-2}$. Novamente, temos $n$ vértices e $n-1$ possíveis graus. Pelo Princípio da Casa dos Pombos, pelo menos dois vértices devem ter o mesmo grau.
5.  Em ambos os casos, é inevitável que pelo menos duas pessoas tenham o mesmo número de conhecidos.

-----

**Questão 4: O complemento de um grafo bipartido é bipartido? Se for, prove, senão, dê um contra-exemplo.**

**Resposta:** Não necessariamente.

**Justificativa (Contra-exemplo):**

1.  [cite\_start]Um grafo é bipartido se, e somente se, não contém ciclos de comprimento ímpar. [cite: 4769]
2.  Considere o grafo $C\_6$ (um ciclo com 6 vértices). $C\_6$ é bipartido, pois não possui ciclos ímpares. Podemos particionar seus vértices em $V\_1 = {v\_1, v\_3, v\_5}$ e $V\_2 = {v\_2, v\_4, v\_6}$. Todas as arestas em $C\_6$ conectam um vértice de $V\_1$ a um de $V\_2$.
3.  Agora, vamos analisar seu complemento, $\\overline{C\_6}$.
4.  Em $\\overline{C\_6}$, dois vértices são adjacentes se não eram em $C\_6$.
5.  Considere os vértices ${v\_1, v\_3, v\_5}$ em $\\overline{C\_6}$.
      * A aresta ${v\_1, v\_3}$ existe em $\\overline{C\_6}$ porque não existe em $C\_6$.
      * A aresta ${v\_3, v\_5}$ existe em $\\overline{C\_6}$ porque não existe em $C\_6$.
      * A aresta ${v\_5, v\_1}$ existe em $\\overline{C\_6}$ porque não existe em $C\_6$.
6.  Essas três arestas formam o ciclo $(v\_1, v\_3, v\_5, v\_1)$ em $\\overline{C\_6}$. Este é um ciclo de comprimento 3, que é ímpar.
7.  Como $\\overline{C\_6}$ possui um ciclo de comprimento ímpar, ele não é bipartido. Portanto, o complemento de um grafo bipartido não é necessariamente bipartido.

-----

**Questão 5: Mostre que $|E(K\_n)| = n(n-1)/2$ e $|E(K\_{p,q})| = p \\cdot q$.**

**Justificativa:**

  * **Para $K\_n$ (Grafo Completo):**

    1.  [cite\_start]Um grafo completo $K\_n$ é um grafo simples com $n$ vértices onde cada par de vértices é conectado por uma aresta. [cite: 1789]
    2.  O número de arestas é o número de maneiras de escolher 2 vértices de um conjunto de $n$ vértices, o que é um problema de combinação.
    3.  A fórmula para combinações de $n$ elementos tomados 2 a 2 é $\\binom{n}{2} = \\frac{n\!}{2\!(n-2)\!} = \\frac{n(n-1)}{2}$.
    4.  Alternativamente, cada um dos $n$ vértices se conecta a outros $n-1$ vértices. Se simplesmente multiplicarmos $n \\times (n-1)$, estaremos contando cada aresta duas vezes (uma para cada ponta). [cite\_start]Portanto, o número de arestas é $\\frac{n(n-1)}{2}$. [cite: 647]

  * **Para $K\_{p,q}$ (Grafo Bipartido Completo):**

    1.  [cite\_start]Um grafo bipartido completo $K\_{p,q}$ tem seu conjunto de vértices particionado em dois conjuntos, $V\_1$ com $p$ vértices e $V\_2$ com $q$ vértices. [cite: 1838]
    2.  Por definição, cada um dos $p$ vértices em $V\_1$ se conecta a *todos* os $q$ vértices em $V\_2$.
    3.  Não há arestas dentro de $V\_1$ ou dentro de $V\_2$.
    4.  Portanto, o número total de arestas é o produto do número de vértices em cada partição: $p \\times q$.

-----

**Questão 6: Seja $G=(V,E)$ um grafo simples com m arestas, quantas arestas contém o grafo $\\overline{G}$?**

**Resposta:** O grafo $\\overline{G}$ contém $\\frac{n(n-1)}{2} - m$ arestas, onde $n=|V|$.

**Justificativa:**

1.  O grafo complementar $\\overline{G}$ tem o mesmo conjunto de vértices $V$ que $G$. [cite\_start]Ele contém todas as arestas que *não* estão em $G$. [cite: 1870]
2.  [cite\_start]O número máximo de arestas que um grafo simples com $n$ vértices pode ter é o número de arestas em um grafo completo $K\_n$, que é $\\frac{n(n-1)}{2}$. [cite: 647]
3.  Se $G$ tem $m$ arestas, então $\\overline{G}$ terá todas as arestas possíveis menos as que já estão em $G$.
4.  Portanto, $|E(\\overline{G})| = |E(K\_n)| - |E(G)| = \\frac{n(n-1)}{2} - m$.

-----

**Questão 7: Seja $G=(V,E)$ um grafo com $|V|=n$ e $|E|=m$. Mostre que se G é um grafo bipartido então $m \\le n^2/4$.**

**Justificativa:**

1.  Seja G um grafo bipartido com $n$ vértices e uma partição de vértices $(V\_1, V\_2)$.
2.  Seja $|V\_1| = p$ e $|V\_2| = q$. Sabemos que $p+q=n$.
3.  O número de arestas em um grafo bipartido é maximizado quando ele é um grafo bipartido completo, ou seja, quando cada vértice em $V\_1$ está conectado a cada vértice em $V\_2$. Nesse caso, o número de arestas é $m = p \\times q$.
4.  Temos que maximizar o produto $p \\times q$ sujeito à restrição $p+q=n$.
5.  Substituindo $q = n-p$, queremos maximizar a função $f(p) = p(n-p) = np - p^2$.
6.  Usando cálculo, a derivada $f'(p) = n - 2p$. Igualando a zero, $n - 2p = 0 \\implies p = n/2$.
7.  Isso significa que o produto é máximo quando as partições são o mais equilibradas possível.
      * Se $n$ é par, $p=q=n/2$, e o número máximo de arestas é $m = (n/2)(n/2) = n^2/4$.
      * Se $n$ é ímpar, as partições serão $\\lfloor n/2 \\rfloor$ e $\\lceil n/2 \\rceil$. Por exemplo, se $n=5$, p=2 e q=3. O produto é $2 \\times 3 = 6$. A fórmula $n^2/4$ dá $25/4 = 6.25$. Portanto, $m \\le n^2/4$ continua válido.
8.  Concluímos que para qualquer grafo bipartido, $m \\le n^2/4$.

-----

**Questão 8:**
**(a) Quais dos seguintes grafos são grafos regulares?**

  * **i. grafos completos:** **Sim**. [cite\_start]Um grafo $K\_n$ é $(n-1)$-regular, pois cada vértice está conectado a todos os outros $n-1$ vértices. [cite: 1662]
  * **ii. ciclos:** **Sim**. [cite\_start]Um ciclo $C\_n$ (com $n \\ge 3$) é 2-regular, pois cada vértice tem exatamente dois vizinhos. [cite: 1673]
  * **iii. grafos bipartidos:** **Não necessariamente**. Um grafo bipartido só é regular se todos os vértices tiverem o mesmo grau. Isso pode acontecer (ex: $C\_6$ é bipartido e 2-regular), mas não é uma regra geral (ex: um caminho $P\_4$ é bipartido, mas tem vértices de grau 1 e 2).
  * **iv. grafos bipartidos completos:** **Apenas se as partições forem de mesmo tamanho**. Em um $K\_{p,q}$, os $p$ vértices de uma partição têm grau $q$, e os $q$ vértices da outra têm grau $p$. Ele só será k-regular se $p=q=k$.

**(b) Quantas arestas possui um grafo k-regular com n vértices? Por quê?**
**Resposta:** $\\frac{n \\times k}{2}$ arestas.
[cite\_start]**Justificativa:** Conforme o Lema do Aperto de Mão, $\\sum d(v) = 2m$. [cite: 547] Em um grafo k-regular com n vértices, cada um dos n vértices tem grau k. Logo, a soma dos graus é $n \\times k$. Portanto, $n \\times k = 2m$, o que implica $m = \\frac{nk}{2}$.

-----

**Questão 9: Os grafos abaixo são isomorfos?**

*Esta questão se refere aos grafos G1 e G2 da lista.*

**Resposta:** Não, os grafos G1 e G2 não são isomorfos.

**Justificativa:**

1.  [cite\_start]Para dois grafos serem isomorfos, eles devem possuir a mesma sequência de graus. [cite: 1446, 1498]
2.  Vamos calcular a sequência de graus para G1 e G2.
      * **G1:**
          * $d(1)=2, d(2)=4, d(3)=2, d(4)=4, d(5)=3, d(6)=3, d(7)=1, d(8)=3$.
          * Sequência de Graus (ordenada): (1, 2, 2, 3, 3, 3, 4, 4).
      * **G2:**
          * $d(a)=3, d(b)=3, d(c)=3, d(d)=3, d(e)=2, d(f)=2$.
          * Primeiro, notamos que G1 tem 8 vértices e G2 tem 6 vértices. Como o número de vértices é diferente, eles **não podem ser isomorfos**.

*Assumindo que a questão tivesse dois grafos com o mesmo número de vértices, a análise da sequência de graus seria o próximo passo. Se mesmo as sequências de graus fossem idênticas, teríamos que procurar por outra invariante, como o número de ciclos de um certo comprimento. Por exemplo, em G2, os vértices (a, c, d, b) formam um ciclo de comprimento 4 ($C\_4$). Teríamos que verificar se G1 possui o mesmo número de $C\_4$ que G2.*

-----

**Questão 10: Mostre que se dois grafos são isomorfos, então possuem o mesmo número de vértices, o mesmo número de arestas, e o grau de cada vértice é preservado. A recíproca é verdadeira?**

**Justificativa:**

1.  **Prova das condições necessárias:**

      * Seja $f: V(G) \\to V(H)$ um isomorfismo entre os grafos G e H.
      * [cite\_start]**Número de Vértices:** Por definição, um isomorfismo é uma **bijeção** entre os conjuntos de vértices. [cite: 1446] Uma bijeção só existe entre conjuntos de mesma cardinalidade. Portanto, $|V(G)| = |V(H)|$.
      * [cite\_start]**Número de Arestas:** A função $f$ preserva adjacência, ou seja, ${u,v} \\in E(G) \\Leftrightarrow {f(u),f(v)} \\in E(H)$. [cite: 1447] Isso cria uma bijeção entre os conjuntos de arestas $E(G)$ e $E(H)$. Portanto, $|E(G)| = |E(H)|$.
      * **Preservação do Grau:** O grau de um vértice $u$ em G é o número de vizinhos que ele tem. Como a bijeção $f$ preserva a adjacência, um vizinho de $u$ em G será mapeado para um vizinho de $f(u)$ em H. Assim, o número de vizinhos é o mesmo, e $d\_G(u) = d\_H(f(u))$ para todo $u \\in V(G)$.

2.  **A recíproca é verdadeira?**

      * **Resposta:** Não.
      * **Contra-exemplo:** Considere os dois grafos abaixo.
          * Grafo A: Duas componentes conexas, cada uma sendo um triângulo ($K\_3$). Vértices: {1,2,3,4,5,6}. Arestas: {(1,2),(2,3),(1,3), (4,5),(5,6),(4,6)}.
          * Grafo B: Um ciclo de 6 vértices ($C\_6$). Vértices: {a,b,c,d,e,f}. Arestas: {(a,b),(b,c),(c,d),(d,e),(e,f),(f,a)}.
      * **Análise:**
          * **Número de Vértices:** Ambos têm 6 vértices.
          * **Número de Arestas:** Ambos têm 6 arestas.
          * **Sequência de Graus:** Em A, todos os 6 vértices têm grau 2. Em B, todos os 6 vértices têm grau 2. A sequência de graus é (2,2,2,2,2,2) para ambos.
      * **Conclusão:** Os grafos possuem as mesmas invariantes, mas não são isomorfos. O Grafo A é desconexo (possui 2 componentes), enquanto o Grafo B é conexo. Conectividade é uma propriedade estrutural que deve ser preservada por um isomorfismo. Como não é, eles não são isomorfos.

-----

**Questão 11: Considerando os grafos apresentados, faça o que se pede:**

**(a) construa a matriz de adjacência do grafo $G\_1$**
[cite\_start]A matriz de adjacência A é uma matriz $n \\times n$ onde $A\_{ij}=1$ se houver uma aresta entre os vértices i e j, e 0 caso contrário. [cite: 2156]

```
   1 2 3 4 5 6 7 8
1 [0,1,1,0,0,0,0,0]
2 [1,0,1,1,1,0,0,0]
3 [1,1,0,0,0,1,0,0]
4 [0,1,0,0,1,1,0,1]
5 [0,1,0,1,0,0,0,1]
6 [0,0,1,1,0,0,1,1]
7 [0,0,0,0,0,1,0,0]
8 [0,0,0,1,1,1,0,0]
```

**(b) construa a matriz de incidência do grafo $G\_2$**
[cite\_start]A matriz de incidência B é uma matriz $n \\times m$ onde $B\_{ij}=1$ se o vértice i é incidente à aresta j. [cite: 2221]
Arestas de G2: $e\_1={a,b}, e\_2={a,c}, e\_3={a,d}, e\_4={b,c}, e\_5={b,d}, e\_6={c,e}, e\_7={d,f}, e\_8={e,f}$.

```
   e1 e2 e3 e4 e5 e6 e7 e8
a [1, 1, 1, 0, 0, 0, 0, 0]
b [1, 0, 0, 1, 1, 0, 0, 0]
c [0, 1, 0, 1, 0, 1, 0, 0]
d [0, 0, 1, 0, 1, 0, 1, 0]
e [0, 0, 0, 0, 0, 1, 0, 1]
f [0, 0, 0, 0, 0, 0, 1, 1]
```

**(c) represente, por meio de uma estrutura de dados, as vizinhanças dos vértices do grafo $G\_1$**
[cite\_start]Usando listas de adjacência. [cite: 2083]

  * $\\Gamma(1) = {2, 3}$
  * $\\Gamma(2) = {1, 3, 4, 5}$
  * $\\Gamma(3) = {1, 2, 6}$
  * $\\Gamma(4) = {2, 5, 6, 8}$
  * $\\Gamma(5) = {2, 4, 8}$
  * $\\Gamma(6) = {3, 4, 7, 8}$
  * $\\Gamma(7) = {6}$
  * $\\Gamma(8) = {4, 5, 6}$

**(d) dê um exemplo de subgrafo em $G\_1$**
[cite\_start]Um subgrafo H de G tem $V(H) \\subseteq V(G)$ e $E(H) \\subseteq E(G)$. [cite: 759]
Exemplo: O grafo induzido pelos vértices ${1, 2, 3}$. Vértices: ${1,2,3}$. Arestas: ${{1,2}, {1,3}, {2,3}}$. Este subgrafo é um triângulo ($K\_3$).

**(e) dê um exemplo de um subgrafo induzido em $G\_2$**
[cite\_start]Um subgrafo induzido por um conjunto de vértices $V'$ contém os vértices de $V'$ e todas as arestas de G com ambos os extremos em $V'$. [cite: 821]
Exemplo: Subgrafo induzido por $V'={a,b,c}$. Vértices: ${a,b,c}$. Arestas: ${{a,b}, {a,c}, {b,c}}$.

**(f) dê um exemplo de um subgrafo gerador em $G\_2$**
[cite\_start]Um subgrafo gerador contém todos os vértices do grafo original. [cite: 805]
Exemplo: O próprio grafo G2, mas removendo a aresta ${a,d}$. Vértices: ${a,b,c,d,e,f}$. Arestas: Todas as arestas de G2 exceto ${a,d}$.

**(g) dê um exemplo de uma clique em $G\_2$**
[cite\_start]Uma clique é um subgrafo que é completo. [cite: 880]
Exemplo: O subgrafo induzido pelos vértices ${a, b, c}$ é uma clique (um $K\_3$), pois todos os seus vértices são mutuamente adjacentes. O subgrafo induzido por ${a, b, d}$ também é uma clique.

-----

**Questão 12: Quantos subgrafos induzidos (por vértice) tem $K\_4$?**

**Resposta:** 15.

**Justificativa:**

1.  [cite\_start]Um subgrafo induzido por vértice é definido unicamente pela escolha de um subconjunto de vértices $V' \\subseteq V$. [cite: 819]
2.  O grafo $K\_4$ tem 4 vértices. O número total de subconjuntos de um conjunto de 4 elementos é $2^4 = 16$.
3.  Os subconjuntos são: 1 subconjunto com 0 vértices (conjunto vazio), 4 subconjuntos com 1 vértice, 6 subconjuntos com 2 vértices, 4 subconjuntos com 3 vértices e 1 subconjunto com 4 vértices.
4.  [cite\_start]A definição de subgrafo induzido por vértices nos slides especifica $V' \\neq \\emptyset$. [cite: 819]
5.  Portanto, excluímos o subconjunto vazio. O número total de subgrafos induzidos não vazios é $16 - 1 = 15$.

-----

**Questão 13: Quantas cliques tem o grafo completo de ordem n?**

**Resposta:** $2^n - 1$.

**Justificativa:**

1.  [cite\_start]Uma clique é um subgrafo que é completo. [cite: 880]
2.  Em um grafo completo $K\_n$, *qualquer* subconjunto de seus vértices induz um subgrafo que também é completo.
3.  Portanto, o número de cliques em $K\_n$ é igual ao número de subconjuntos não vazios de seus vértices.
4.  Um conjunto de $n$ vértices tem $2^n$ subconjuntos no total.
5.  Excluindo o subconjunto vazio, temos $2^n - 1$ cliques.

-----

**Questão 14: Considere o seguinte conjunto de 10 peças de dominó... Mostre (usando grafos) a possibilidade de arranjar as peças numa sequência.**

**Resposta:** Sim, é possível.

**Justificativa:**

1.  Modelamos este problema usando um grafo $G$. Os vértices são os números que aparecem nas peças: $V = {1, 2, 3, 4, 5}$.
2.  Cada peça de dominó, como (1,2), representa uma aresta entre os vértices correspondentes, {1,2}.
3.  [cite\_start]O problema de arranjar as peças em sequência, de forma que os números se conectem, é equivalente a encontrar um **caminho euleriano** no grafo, que é um caminho que percorre cada aresta exatamente uma vez. [cite: 2581]
4.  [cite\_start]Um grafo conexo possui um caminho euleriano se, e somente se, ele tiver 0 ou 2 vértices de grau ímpar. [cite: 2564-2581] (Implícito no slide sobre linha de Euler e nos teoremas de grafos eulerianos).
5.  Vamos calcular os graus de cada vértice no nosso grafo:
      * $d(1)$: aparece nas peças (1,2), (1,3), (1,4), (1,5). Grau = 4.
      * $d(2)$: aparece nas peças (1,2), (2,3), (2,4), (2,5). Grau = 4.
      * $d(3)$: aparece nas peças (1,3), (2,3), (3,4), (3,5). Grau = 4.
      * $d(4)$: aparece nas peças (1,4), (2,4), (3,4), (4,5). Grau = 4.
      * $d(5)$: aparece nas peças (1,5), (2,5), (3,5), (4,5). Grau = 4.
6.  Todos os 5 vértices têm grau 4, que é um grau par. O número de vértices de grau ímpar é 0.
7.  [cite\_start]Como o grafo é conexo e todos os seus vértices têm grau par, ele é um **grafo euleriano**. [cite: 2653] Isso significa que ele não apenas possui um caminho euleriano, mas também um **ciclo euleriano** (uma linha de Euler).
8.  Portanto, é possível arranjar todas as peças em uma sequência e, além disso, a sequência pode ser fechada, com a última peça se conectando à primeira.

-----

**Questão 15: É verdade que todo grafo 2-regular é um ciclo? Justifique sua resposta.**

**Resposta:** Não.

**Justificativa:**

1.  [cite\_start]Um grafo 2-regular é aquele onde todos os vértices têm grau 2. [cite: 1662]
2.  Um ciclo é, de fato, um exemplo de grafo 2-regular conexo.
3.  No entanto, um grafo 2-regular não precisa ser conexo.
4.  **Contra-exemplo:** Considere um grafo G formado pela **união disjunta de dois ou mais ciclos**. Por exemplo, um grafo com 6 vértices composto por dois subgrafos de triângulo ($K\_3$) separados.
5.  Nesse grafo, cada um dos 6 vértices tem grau 2, então ele é 2-regular. No entanto, ele não é um único ciclo, mas sim duas componentes conexas, cada uma sendo um ciclo. Portanto, nem todo grafo 2-regular é um ciclo.

-----

### **Lista de Exercícios 2**

*Devido à natureza gráfica de algumas questões, as respostas serão descritivas.*

**Questão 1: Defina um grafo ... de 7 vértices e 9 arestas com sequencia de graus 2342331. Exiba...**

**Resposta:**
Primeiro, verificamos a viabilidade. [cite\_start]Soma dos graus = 2+3+4+2+3+3+1 = 18. Como $2m = 18 \\implies m=9$, a sequência é plausível. [cite: 547]
**Grafo Exemplo:**

  * Vértices: $V={1,2,3,4,5,6,7}$ com graus $d(1)=2, d(2)=3, d(3)=4, d(4)=2, d(5)=3, d(6)=3, d(7)=1$.
  * Arestas: ${1,2}, {1,3}, {2,3}, {2,5}, {3,4}, {3,5}, {4,5}, {5,6}, {6,7}$.
  * (A construção de um grafo específico pode variar, mas este satisfaz as condições).

**Exibindo os percursos:**

  * **(a) percurso aberto de comprimento 10:** Um percurso pode repetir vértices e arestas. Ex: `7-6-5-4-3-5-2-3-1-2-5`
  * [cite\_start]**(b) um ciclo:** Um percurso simples e fechado. [cite: 5305] Ex: `1-2-3-1`.
  * [cite\_start]**(c) um caminho de comprimento 6:** Um percurso que não repete vértices. [cite: 5289] Ex: `7-6-5-2-1-3-4`.
  * [cite\_start]**(d) um ciclo elementar:** Um ciclo onde só o primeiro e último vértice se repetem. [cite: 5320] Ex: `2-3-5-2`.
  * **(e) um percurso fechado que não seja simples nem elementar:** Deve repetir arestas. Ex: `1-2-3-1-2-3-1`.

-----

**Questão 2: Todo percurso elementar é simples. Todo percurso simples é elementar? Explique.**

**Resposta:**

1.  **Todo percurso elementar é simples:** **Sim**. [cite\_start]Um percurso elementar não repete vértices. [cite: 5289] Se um percurso não repete vértices, ele não pode repetir arestas, pois para repetir uma aresta ${u,v}$, seria necessário passar por u, ir para v, e depois voltar para u (ou vice-versa), o que repetiria um vértice. Portanto, a definição de elementar é mais restritiva e implica a de simples.

2.  **Todo percurso simples é elementar?** **Não**. [cite\_start]Um percurso simples não repete arestas. [cite: 5275] No entanto, ele pode repetir vértices.

      * **Exemplo:** Considere os vértices A, B, C e as arestas ${A,B}$ e ${B,C}$. O percurso `A-B-C-B` é **simples**, pois não repete arestas (usou ${A,B}$ e depois ${B,C}$). Contudo, ele **não é elementar**, pois o vértice B foi repetido.

-----

**Questão 3: Dê um exemplo de um grafo simples e conexo que não possua ciclos de comprimento ímpar.**

**Resposta:**
[cite\_start]Um grafo é bipartido se, e somente se, não contém ciclo ímpar. [cite: 4769] Portanto, qualquer grafo bipartido conexo serve como exemplo.

  * **Exemplo 1:** Um ciclo de comprimento par, como $C\_4$ ou $C\_6$.
  * **Exemplo 2:** Um grafo bipartido completo, como $K\_{3,3}$.
  * **Exemplo 3:** Uma árvore. Toda árvore é um grafo bipartido e, por definição, não possui ciclos de nenhum tipo.

-----

**Questão 4: Explique por que se um grafo (conexo ou desconexo) tem exatamente dois vértices de grau ímpar, então existe um caminho que liga esses dois vértices.**

[cite\_start]**Justificativa (Propriedade 4):** [cite: 4755]

1.  [cite\_start]A prova dessa propriedade está relacionada à estrutura das componentes conexas e ao corolário de que o número de vértices de grau ímpar é sempre par. [cite: 583]
2.  Vamos considerar uma componente conexa qualquer de um grafo. Dentro dessa componente, a soma dos graus de seus vértices deve ser par. Isso implica que o número de vértices de grau ímpar *dentro de qualquer componente conexa* também deve ser par.
3.  Se o grafo inteiro tem exatamente dois vértices de grau ímpar, $u$ e $v$, eles não podem pertencer a componentes conexas diferentes. Se pertencessem, a componente de $u$ teria apenas um vértice de grau ímpar (o próprio $u$), e a componente de $v$ também teria apenas um (o próprio $v$), o que é impossível.
4.  Portanto, $u$ e $v$ devem, obrigatoriamente, pertencer à **mesma componente conexa**.
5.  [cite\_start]Por definição de componente conexa, se dois vértices estão na mesma componente, existe um caminho que os liga. [cite: 4508]

-----

**Questão 5: Mostre que um grafo simples com n vértices e mais que $(n-1)(n-2)/2$ arestas é conexo.**

**Justificativa (Prova por Contradição):**

1.  Vamos supor que um grafo $G$ com $n$ vértices e $m \> \\frac{(n-1)(n-2)}{2}$ arestas **seja desconexo**.
2.  [cite\_start]Um grafo desconexo tem pelo menos duas componentes conexas ($k \\ge 2$). [cite: 4522]
3.  [cite\_start]Para maximizar o número de arestas em um grafo desconexo com $n$ vértices, devemos fazer as componentes o mais "densas" possível, ou seja, completas. [cite: 4908]
4.  A configuração que maximiza o número de arestas em um grafo desconexo é ter uma componente sendo um grafo completo $K\_{n-1}$ e a outra componente sendo um vértice isolado.
5.  O número de arestas em $K\_{n-1}$ é $\\frac{(n-1)((n-1)-1)}{2} = \\frac{(n-1)(n-2)}{2}$.
6.  Este é o número máximo de arestas que um grafo simples de $n$ vértices pode ter e ainda assim ser desconexo.
7.  Se o nosso grafo $G$ tem $m \> \\frac{(n-1)(n-2)}{2}$ arestas, ele tem mais arestas do que o máximo possível para um grafo desconexo. Essa aresta adicional deve conectar as componentes (neste caso, o vértice isolado ao $K\_{n-1}$), tornando o grafo conexo.
8.  Isso contradiz nossa suposição inicial. Portanto, o grafo deve ser conexo.

-----

**Questão 6: Mostre que um grafo simples G permanece conexo mesmo depois da remoção de uma aresta a de G se e somente se a pertence a algum ciclo de G.**

**Justificativa:**

  * **Parte 1 (⇒): Se G-a é conexo, então a pertence a um ciclo.**

    1.  Seja $a = {u,v}$ a aresta removida.
    2.  [cite\_start]Como $G-a$ ainda é conexo, deve existir um caminho entre $u$ e $v$ em $G-a$. [cite: 4297]
    3.  Seja $P$ esse caminho de $u$ para $v$ em $G-a$.
    4.  Ao adicionarmos a aresta $a={u,v}$ de volta ao grafo, o caminho $P$ junto com a aresta $a$ forma um ciclo $(u \\to ... \\to v \\to u)$.
    5.  Portanto, a aresta $a$ pertence a um ciclo em G.

  * **Parte 2 (⇐): Se a pertence a um ciclo, então G-a é conexo.**

    1.  Seja G um grafo conexo e seja $a = {u,v}$ uma aresta que pertence a um ciclo $C$.
    2.  Vamos provar que $G-a$ é conexo. Precisamos mostrar que para quaisquer dois vértices $x,y \\in V$, existe um caminho entre eles em $G-a$.
    3.  Como G é conexo, existe um caminho $P$ entre $x$ e $y$ em G.
    4.  **Caso A:** Se o caminho $P$ não utiliza a aresta $a$, então esse mesmo caminho $P$ existe em $G-a$, e $x$ e $y$ estão conectados.
    5.  **Caso B:** Se o caminho $P$ utiliza a aresta $a$. Podemos percorrer o ciclo $C$ no sentido oposto ao da aresta $a$ para ir de $u$ para $v$ (ou de $v$ para $u$). Seja $C'$ essa parte do ciclo. $C'$ é um caminho alternativo entre $u$ e $v$ que não usa a aresta $a$. Podemos substituir a aresta $a$ no caminho $P$ pelo caminho $C'$, criando um novo percurso de $x$ para $y$ que existe inteiramente em $G-a$.
    6.  Em ambos os casos, existe um caminho entre $x$ e $y$ em $G-a$. Portanto, $G-a$ é conexo.

-----

**Questão 7: Uma aresta a de um grafo G é uma ponte se e somente se G-a é desconexo. (a) Dê exemplo de um grafo conexo simples que não tenha pontes.**

**Resposta:**
[cite\_start]A definição de ponte é exatamente uma aresta cuja remoção desconecta o grafo. [cite: 2748] A questão 6 nos diz que uma aresta não é uma ponte se e somente se ela pertence a um ciclo. Portanto, para um grafo não ter pontes, todas as suas arestas devem pertencer a algum ciclo.

  * **Exemplo:** Qualquer grafo de ciclo, como $C\_3, C\_4, C\_5, ...$. Em um ciclo, a remoção de qualquer aresta resulta em um caminho, que ainda é um grafo conexo. Outro exemplo é qualquer grafo completo $K\_n$ com $n \\ge 3$.

-----

**Questão 8: Dê exemplos de:**

  * **a) um digrafo que seja f-conexo.**
    [cite\_start]Um digrafo é f-conexo (fortemente conexo) se todo par de vértices é mutuamente atingível. [cite: 6200]
    **Exemplo:** Um ciclo direcionado, como $V={1,2,3}, A={(1,2), (2,3), (3,1)}$.

  * **b) um digrafo que seja sf-conexo e que não seja f-conexo.**
    [cite\_start]Um digrafo é sf-conexo (semi-fortemente) se para todo par $u,v$, ou $u$ atinge $v$ ou $v$ atinge $u$. [cite: 6181]
    **Exemplo:** Um caminho direcionado, como $V={1,2,3}, A={(1,2), (2,3)}$. O vértice 1 atinge 3, mas 3 não atinge 1. Não é f-conexo. Mas para qualquer par, há um caminho em pelo menos uma direção.

  * **c) um digrafo que seja acíclico mas o seu grafo subjacente possua ciclos.**
    **Exemplo:** $V={1,2,3}, A={(1,2), (1,3), (2,3)}$. Este digrafo é acíclico (DAG). [cite\_start]Seu grafo subjacente possui as arestas ${1,2}, {1,3}, {2,3}$, formando um ciclo (triângulo). [cite: 6096, 6110]

  * **d) Um grafo orientado simplesmente conexo que possua pelo menos 3 componentes f-conexas.**
    [cite\_start]Um digrafo é s-conexo se seu grafo subjacente for conexo. [cite: 6180] [cite\_start]Componentes f-conexas são subgrafos maximais f-conexos. [cite: 6400]
    **Exemplo:** Considere três ciclos direcionados $C\_1={v\_1,v\_2}, C\_2={v\_3,v\_4}, C\_3={v\_5,v\_6}$. Estes são 3 componentes f-conexas. Para torná-lo s-conexo, basta adicionar arcos que conectem as componentes, como $(v\_2, v\_3)$ e $(v\_4, v\_5)$. O grafo subjacente será conexo, mas as componentes f-conexas permanecem separadas.

-----

**Questão 9: Dado o grafo da Figura 1, determine:**

**(a) os graus de entrada dos vértices**
$d^-(a)=1, d^-(b)=1, d^-(c)=2, d^-(d)=1, d^-(e)=1, d^-(f)=1, d^-(g)=1, d^-(h)=1$.

**(b) os graus de saída dos vértices**
$d^+(a)=1, d^+(b)=2, d^+(c)=2, d^+(d)=1, d^+(e)=2, d^+(f)=1, d^+(g)=2, d^+(h)=0$.

**(c) o fecho transitivo direto do vértice a**
[cite\_start]O fecho transitivo direto $\\hat{\\Gamma}^+(a)$ é o conjunto de vértices atingíveis a partir de a. [cite: 5910]

  * De a, podemos ir para b.
  * De b, podemos ir para e e c.
  * De e, podemos ir para c e f.
  * De c, podemos ir para d e f.
  * De f, podemos ir para g.
  * De g, podemos ir para e e h.
  * De d, podemos ir para a.
  * Atingíveis: {a, b, c, d, e, f, g, h}. Portanto, $\\hat{\\Gamma}^+(a) = V$.

**(d) o fecho transitivo inverso do vértice h**
[cite\_start]O fecho transitivo inverso $\\hat{\\Gamma}^-(h)$ é o conjunto de vértices que podem atingir h. [cite: 6006]

  * h é atingido por g.
  * g é atingido por f e c.
  * f é atingido por e e c.
  * e é atingido por b e g.
  * c é atingido por b e e.
  * b é atingido por a.
  * a é atingido por d.
  * d é atingido por c.
  * Todos os vértices podem, eventualmente, atingir h. Portanto, $\\hat{\\Gamma}^-(h) = V$.

**(e) os sucessores do vértice b**
Os sucessores são os vértices adjacentes de saída. [cite\_start]$\\Gamma^+(b) = {c, e}$. [cite: 5832]

**(f) os antecessores do vértice c**
Os antecessores são os vértices adjacentes de entrada. [cite\_start]$\\Gamma^-(c) = {b, e}$. [cite: 5830]

-----

**Questão 10: Dê exemplo de um grafo conexo simples que só possua pontes.**
**Resposta:** Qualquer árvore. Uma árvore é um grafo conexo acíclico. [cite\_start]Como não há ciclos, a remoção de qualquer aresta necessariamente desconectará o grafo (ou uma parte dele), tornando cada aresta uma ponte. [cite: 2748]
**Exemplo concreto:** Um caminho $P\_n$ ou uma estrela $K\_{1,n-1}$.

-----

**Questão 11: Prove que um grafo simples G com n vértices e k componentes conexas pode ter no máximo $(n-k)(n-k+1)/2$ arestas.**
[cite\_start]Esta é exatamente a **Propriedade 6** demonstrada nos slides de Conexidade. [cite: 4891] A demonstração se baseia em maximizar o número de arestas, o que ocorre quando as componentes são completas. A prova, conforme os slides, usa indução ou a desigualdade de Cauchy-Schwarz para mostrar que a soma $\\sum (n\_i)^2$ é maximizada quando uma componente tem $n-k+1$ vértices e as outras $k-1$ componentes têm 1 vértice cada. [cite\_start]Isso leva ao resultado final. [cite: 4906-4982]

-----

**Questão 12: Implementar o algoritmo de busca em profundidade para:**
**(a) determinar componentes conexas de um grafo não orientado.**
[cite\_start]O pseudocódigo para isso está nos slides de "Caminhamento em Grafos". [cite: 6933, 6948] A lógica é:

1.  Inicialize um contador de componentes `comp = 0` e marque todos os vértices como não visitados.
2.  Itere por todos os vértices de $v=1$ a $n$.
3.  Se o vértice $v$ não foi visitado, incremente `comp`, e inicie uma DFS a partir de $v$, marcando todos os vértices alcançáveis com o número da componente atual (`comp`).
4.  Ao final, todos os vértices pertencentes à mesma componente terão a mesma marcação.

**(b) determinar as componentes f-conexas de um digrafo.**
A determinação de componentes f-conexas (SCCs) é mais complexa. Um algoritmo clássico (não detalhado nos slides, mas cuja aplicação é pedida) é o de Kosaraju ou o de Tarjan. O de Kosaraju usa duas passagens de DFS:

1.  Execute uma DFS no digrafo G para calcular os "tempos de finalização" de cada vértice.
2.  Calcule o grafo reverso (transposto) $G^T$.
3.  Execute uma DFS em $G^T$, processando os vértices em ordem decrescente de seus tempos de finalização calculados na etapa 1.
4.  Cada árvore na floresta de DFS da segunda busca corresponde a uma componente fortemente conexa.

-----

**Questão 13: Implementar o algoritmo de busca em largura e aplicá-lo ao grafo $G\_1$... a partir do vértice 1... Forneça a árvore de visitas do grafo. O que ela representa?**
[cite\_start]**Algoritmo de Busca em Largura (BFS):** [cite: 7011]
**Aplicação em G1 a partir do vértice 1:**

1.  Fila Q = [1]. Distâncias d[1]=0.
2.  Retira 1. Vizinhos não visitados: 2, 3. Adiciona na fila. Q=[2, 3]. d[2]=1, d[3]=1. Ant[2]=1, Ant[3]=1.
3.  Retira 2. Vizinhos não visitados: 4, 5. Adiciona na fila. Q=[3, 4, 5]. d[4]=2, d[5]=2. Ant[4]=2, Ant[5]=2.
4.  Retira 3. Vizinho não visitado: 6. Adiciona na fila. Q=[4, 5, 6]. d[6]=2. Ant[6]=3.
5.  Retira 4. Vizinho não visitado: 8. Adiciona na fila. Q=[5, 6, 8]. d[8]=3. Ant[8]=4.
6.  Retira 5. Nenhum vizinho novo. Q=[6, 8].
7.  Retira 6. Vizinho não visitado: 7. Adiciona na fila. Q=[8, 7]. d[7]=3. Ant[7]=6.
8.  Retira 8. Nenhum vizinho novo. Q=[7].
9.  Retira 7. Nenhum vizinho novo. Fila vazia. Fim.

**Árvore de Visitas:**

  * Raiz: 1
  * Nível 1 (filhos de 1): 2, 3
  * Nível 2 (filhos de 2 e 3): 4, 5 (de 2); 6 (de 3)
  * Nível 3 (filhos de 4, 5, 6): 8 (de 4); 7 (de 6)

**O que ela representa?**
[cite\_start]A árvore de visitas da BFS representa a **árvore de caminhos mínimos** (em termos de número de arestas) da origem (vértice 1) para todos os outros vértices alcançáveis. [cite: 6993] A distância de qualquer nó para a raiz na árvore é o menor número de arestas necessárias para chegar a esse nó a partir da raiz no grafo original.

-----

### **Lista de Exercícios 3**

**Questão 1: Existe um grafo bipartido hamiltoniano com número ímpar de vértices?**

**Resposta:** Não.

**Justificativa:**

1.  [cite\_start]Um ciclo hamiltoniano é um ciclo que visita cada vértice do grafo exatamente uma vez. [cite: 2836]
2.  [cite\_start]Em um grafo bipartido, os vértices são divididos em duas partições, $V\_1$ e $V\_2$, e as arestas só existem entre as partições. [cite: 1821]
3.  Qualquer ciclo em um grafo bipartido deve alternar entre vértices de $V\_1$ e $V\_2$ (ex: $v\_1 \\in V\_1 \\to v\_2 \\in V\_2 \\to v\_3 \\in V\_1 \\to ...$).
4.  Para fechar o ciclo e voltar ao vértice inicial, o número total de "passos" (arestas) deve ser par. Isso implica que um ciclo em um grafo bipartido sempre tem um número par de vértices.
5.  Como um ciclo hamiltoniano deve incluir todos os vértices do grafo, se um grafo bipartido for hamiltoniano, seu número total de vértices ($n = |V\_1| + |V\_2|$) deve ser par.
6.  Portanto, um grafo bipartido com um número ímpar de vértices não pode ser hamiltoniano.

-----

**Questão 2: Se um grafo bipartido $G=(V,E)$, com bipartição $V=(A,B)$ possui caminho hamiltoniano então $|A|=|B|$. Verdadeiro ou falso?**

**Resposta:** Falso.

**Justificativa (Contra-exemplo):**

1.  [cite\_start]Um caminho hamiltoniano visita cada vértice exatamente uma vez. [cite: 2839]
2.  Assim como um ciclo, um caminho em um grafo bipartido deve alternar entre as partições A e B.
3.  Se um caminho começa em A e termina em A, ele deve ter um número ímpar de vértices ($|A| = |B|+1$). Ex: `A-B-A-B-A`.
4.  Se um caminho começa em A e termina em B, ele deve ter um número par de vértices ($|A| = |B|$). Ex: `A-B-A-B`.
5.  Portanto, a condição $|A|=|B|$ só é necessária se o caminho tiver um número par de vértices. Se o caminho tiver um número ímpar de vértices, as partições devem ter tamanhos que diferem em 1.
6.  **Contra-exemplo:** Considere o grafo caminho $P\_5$ com vértices ${1,2,3,4,5}$. Ele é bipartido com partições $A={1,3,5}$ e $B={2,4}$. Aqui, $|A|=3$ e $|B|=2$. O próprio grafo é um caminho hamiltoniano, mas $|A| \\ne |B|$.

-----

**Questão 3: Considere o grafo da Figura 1 e responda:**

*(As respostas são baseadas no grafo G1 da lista, que é o mesmo da lista 1)*

  * **(a) Determine a distância $d(v,w)$ entre cada par de vértice.**
    [cite\_start]A distância é o caminho mais curto (menor número de arestas). [cite: 6560] O cálculo é feito executando BFS a partir de cada vértice. A matriz de distâncias D será:

    ```
       1 2 3 4 5 6 7 8
    1 [0,1,1,2,2,2,3,3]
    2 [1,0,1,1,1,2,3,2]
    3 [1,1,0,2,2,1,2,2]
    4 [2,1,2,0,1,1,2,1]
    5 [2,1,2,1,0,2,3,1]
    6 [2,2,1,1,2,0,1,1]
    7 [3,3,2,2,3,1,0,2]
    8 [3,2,2,1,1,1,2,0]
    ```

  * **(b) Qual é a excentricidade de cada vértice?**
    [cite\_start]A excentricidade $E(v)$ é o valor máximo na linha correspondente de $v$ na matriz de distâncias. [cite: 6584]

      * $E(1)=3, E(2)=3, E(3)=2, E(4)=2, E(5)=3, E(6)=2, E(7)=3, E(8)=3$.

  * **(c) Qual é o raio do grafo?**
    [cite\_start]O raio é a menor das excentricidades. [cite: 6598]
    Raio = $\\min(3,3,2,2,3,2,3,3) = 2$.

  * **(d) Qual é o centro do grafo?**
    [cite\_start]O centro é o conjunto de vértices com excentricidade mínima (igual ao raio). [cite: 6611]
    Centro = ${3, 4, 6}$.

  * **(e) Qual é o diâmetro do grafo?**
    [cite\_start]O diâmetro é a maior das excentricidades. [cite: 6625]
    Diâmetro = $\\max(3,3,2,2,3,2,3,3) = 3$.

  * **(f) Determine quais são os vértices periféricos.**
    [cite\_start]Vértices periféricos são aqueles cuja excentricidade é igual ao diâmetro. [cite: 6626]
    Vértices Periféricos = ${1, 2, 5, 7, 8}$.

  * **(g) explique com suas palavras como seria um algoritmo para calcular o diâmetro de um grafo.**

    1.  Inicialize uma variável `diametro_max = 0`.
    2.  Para cada vértice `v` no grafo:
        a. Calcule a distância de `v` para todos os outros vértices `w` do grafo. Isso pode ser feito usando uma Busca em Largura (BFS) partindo de `v`.
        b. Encontre a maior dessas distâncias. Este valor é a excentricidade de `v`, $E(v)$.
        c. Compare $E(v)$ com `diametro_max`. Se $E(v) \> diametro\_max$, atualize `diametro_max = E(v)`.
    3.  Após iterar por todos os vértices, o valor final de `diametro_max` será o diâmetro do grafo. Essencialmente, é um algoritmo "Todos-os-Pares-Caminhos-Mais-Curtos" seguido da busca pelo máximo dos máximos.

-----

**Questão 4: Por que o algoritmo de Dijkstra não garante resultados corretos para grafos com arestas negativas? Mostre um exemplo.**

**Justificativa:**

1.  O algoritmo de Dijkstra é um algoritmo "guloso". [cite\_start]Ele opera sob a premissa de que, uma vez que um vértice `k` é selecionado (por ter o menor custo temporário) e movido para o conjunto "fechado", o caminho encontrado para ele é definitivamente o mais curto. [cite: 3870-3884]
2.  Essa premissa só é válida se os pesos das arestas forem não-negativos. Se forem não-negativos, qualquer outro caminho para `k` que passe por um vértice ainda "aberto" (com custo maior) nunca poderá resultar em um custo total menor.
3.  Com arestas negativas, essa lógica falha. É possível que, após fechar um vértice `k`, se descubra um caminho mais curto para ele através de outro vértice `j` (ainda aberto), se o caminho de `j` para `k` envolver uma aresta negativa suficientemente grande. [cite\_start]O algoritmo de Dijkstra, por não reabrir vértices, nunca descobrirá esse caminho melhor. [cite: 4212]

**Exemplo:**

  * [cite\_start]Considere o grafo do slide "TG08-GrafosAula8-Dijkstra", página 16. [cite: 4212]
  * Vértices: {1, 2, 3}. Arestas: $(1,2)$ com peso 10, $(1,3)$ com peso 3, $(2,3)$ com peso -8.
  * **Execução de Dijkstra a partir de 1:**
    1.  d[1]=0, d[2]=inf, d[3]=inf.
    2.  Abre 1. Vizinhos: 2 e 3. Atualiza d[2]=10, d[3]=3.
    3.  Fecha 3 (o menor, custo 3). O algoritmo declara que o caminho mais curto para 3 é 3.
    4.  Abre 2. Vizinhos: 3. Testa o caminho $1 \\to 2 \\to 3$. Custo = $d(2) + c(2,3) = 10 + (-8) = 2$.
    5.  Como $2 \< d(3)$, o custo de 3 deveria ser 2. Mas como o vértice 3 já foi "fechado", Dijkstra não o reavalia.
  * **Resultado Final de Dijkstra:** Caminho mais curto para 3 é 3.
  * **Resultado Correto:** Caminho mais curto para 3 é $1 \\to 2 \\to 3$, com custo 2.

-----

**Questão 5: Explique por que grafos com ciclos negativos são particularmente problemáticos para os algoritmos de caminho mínimo.**

**Justificativa:**

1.  Um ciclo de comprimento negativo é um ciclo cuja soma dos pesos das arestas é menor que zero.
2.  [cite\_start]Se existe um caminho de um vértice `s` para um vértice `t` que passa por um ciclo negativo, não existe um "caminho mais curto" bem definido. [cite: 3818]
3.  A razão é que podemos percorrer o ciclo negativo repetidamente. Cada vez que o ciclo é percorrido, o custo total do caminho de `s` para `t` diminui.
4.  [cite\_start]Podemos fazer isso um número infinito de vezes, fazendo o custo do caminho tender a $-\\infty$. [cite: 3811]
5.  Portanto, o problema em si se torna ilimitado e não tem uma solução finita. [cite\_start]Algoritmos como Bellman-Ford são capazes de detectar a presença desses ciclos, mas não podem fornecer um caminho de custo mínimo nesses casos. [cite: 3394]

-----

**Questão 6: Usando o algoritmo de Dijkstra ache os caminhos mínimos do vértice A (no slide está como 1) para todos os outros vértices do grafo da Figura 2.**

*A questão na lista pede do vértice 1, mas o grafo na lista é rotulado com letras. Vamos assumir que a questão se refere ao grafo da Figura 2 e o vértice inicial é 'A'.*

**Execução do Algoritmo de Dijkstra (Origem: A):**

  * **Inicialização:**

      * `aberto` = {A,B,C,D,E,F,G,H,I}
      * `fechado` = {}
      * `d(A)=0`, `d(X)=inf` para todo X \!= A
      * `anterior(X)=null` para todo X.

  * **Iteração 1:**

      * `k=A` (o mais próximo da origem). `fechado`={A}.
      * Vizinhos de A: C (custo=9), D (custo=12), B (custo=22).
      * Atualiza: `d(C)=9`, `ant(C)=A`; `d(D)=12`, `ant(D)=A`; `d(B)=22`, `ant(B)=A`.

  * **Iteração 2:**

      * `k=C` (o mais próximo em `aberto`, com custo 9). `fechado`={A,C}.
      * Vizinhos de C: B (d(A)+c(A,C)+c(C,B) = 9+35=44 \> d(B)=22. Não atualiza). F (d(C)+c(C,F)=9+42=51). E (d(C)+c(C,E)=9+65=74). D (d(C)+c(C,D)=9+4=13 \> d(D)=12. Não atualiza).
      * Atualiza: `d(F)=51`, `ant(F)=C`; `d(E)=74`, `ant(E)=C`.

  * **Iteração 3:**

      * `k=D` (custo 12). `fechado`={A,C,D}.
      * Vizinhos de D: E (d(D)+c(D,E)=12+33=45 \< d(E)=74. Atualiza). I (d(D)+c(D,I)=12+30=42).
      * Atualiza: `d(E)=45`, `ant(E)=D`; `d(I)=42`, `ant(I)=D`.

  * **Iteração 4:**

      * `k=B` (custo 22). `fechado`={A,C,D,B}.
      * Vizinhos de B: F (d(B)+c(B,F)=22+36=58 \> d(F)=51. Não atualiza). H (d(B)+c(B,H)=22+34=56).
      * Atualiza: `d(H)=56`, `ant(H)=B`.

  * **Iteração 5:**

      * `k=I` (custo 42). `fechado`={A,C,D,B,I}.
      * Vizinhos de I: G (d(I)+c(I,G)=42+21=63). H (d(I)+c(I,H)=42+19=61 \> d(H)=56. Não atualiza).
      * Atualiza: `d(G)=63`, `ant(G)=I`.

  * **Iteração 6:**

      * `k=E` (custo 45). `fechado`={A,C,D,B,I,E}.
      * Vizinhos de E: F (d(E)+c(E,F)=45+18=63 \> d(F)=51. Não atualiza). G (d(E)+c(E,G)=45+23=68 \> d(G)=63. Não atualiza).

  * **Iteração 7:**

      * `k=F` (custo 51). `fechado`={A,C,D,B,I,E,F}.
      * Vizinhos de F: H (d(F)+c(F,H)=51+24=75 \> d(H)=56. Não atualiza). G (d(F)+c(F,G)=51+39=90 \> d(G)=63. Não atualiza).

  * **Iteração 8:**

      * `k=H` (custo 56). `fechado`={A,C,D,B,I,E,F,H}.

  * **Iteração 9:**

      * `k=G` (custo 63). `fechado`={A,C,D,B,I,E,F,H,G}. Fim.

**Resultados Finais (Caminhos e Custos):**

  * **A -\> C:** A-C (Custo: 9)
  * **A -\> D:** A-D (Custo: 12)
  * **A -\> B:** A-B (Custo: 22)
  * **A -\> I:** A-D-I (Custo: 42)
  * **A -\> E:** A-D-E (Custo: 45)
  * **A -\> F:** A-C-F (Custo: 51)
  * **A -\> H:** A-B-H (Custo: 56)
  * **A -\> G:** A-D-I-G (Custo: 63)

-----

**Questão 7: Desenhe:**

**(a) um grafo euleriano e hamiltoniano;**
Um ciclo $C\_n$ com $n \\ge 3$. Ele é 2-regular (logo, euleriano) e é seu próprio ciclo hamiltoniano. Ex: $C\_5$.

**(b) um grafo euleriano e não hamiltoniano;**
Exemplo: Duas cópias de $K\_3$ (triângulos) que compartilham um único vértice. Todos os vértices têm grau par (2 ou 4), então é euleriano. Mas não é hamiltoniano, pois o vértice de junção teria que ser visitado duas vezes para visitar todos os outros vértices.

**(c) um grafo não euleriano e hamiltoniano;**
O grafo completo $K\_4$ com uma aresta removida. Ele ainda possui um ciclo hamiltoniano. No entanto, ele terá dois vértices de grau 2 e dois vértices de grau 3. Como tem vértices de grau ímpar, não é euleriano.

**(d) um grafo não euleriano e não hamiltoniano.**
Um grafo de "haltere": dois triângulos conectados por um caminho. Ele tem vértices de grau ímpar (os extremos do caminho), então não é euleriano. Ele também não é hamiltoniano porque as pontes (arestas do caminho) teriam que ser percorridas duas vezes para visitar todos os vértices em um ciclo, o que não é permitido.
