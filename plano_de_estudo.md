

## **Guia de Estudo Completo de Teoria dos Grafos**

Este documento foi elaborado a partir do cruzamento das suas listas de exercícios e do material de aula, com o objetivo de fornecer um guia direcionado para a resolução dos problemas propostos.

### **Tópico 1: Definições e Conceitos Fundamentais**

Esses conceitos são a base para a maioria dos exercícios, especialmente os da **Lista 1** e **Lista 2**.

#### **1.1. Estrutura de um Grafo**
* [cite_start]**Definição:** Um grafo $G$ é um par de conjuntos $(V, E)$, onde $V$ é o conjunto de vértices (ou nós) e $E$ é o conjunto de arestas (ou arcos)[cite: 454, 465].
* [cite_start]Uma aresta $e_k$ é definida por um par de vértices $\{v_i, v_j\}$, que são seus extremos[cite: 482].
* [cite_start]**Ordem e Tamanho:** A ordem de um grafo é seu número de vértices, $n = |V|$[cite: 671]. [cite_start]O tamanho é seu número de arestas, $m = |E|$[cite: 684].

#### **1.2. Tipos de Grafos**
* [cite_start]**Multigrafo:** Um grafo que permite a existência de **arestas paralelas** (arestas que possuem os mesmos vértices extremos) e **laços** (arestas do tipo $\{v_i, v_i\}$)[cite: 579, 594, 565].
* [cite_start]**Grafo Simples:** Um grafo que **não possui** arestas paralelas nem laços[cite: 611]. A maioria dos teoremas e exercícios se aplicam a grafos simples.

#### **1.3. Adjacência e Incidência**
* [cite_start]**Incidência:** Vértices são incidentes a uma aresta se forem seus extremos, e a aresta é incidente a eles[cite: 732, 733]. [cite_start]É uma relação entre vértices e arestas[cite: 775].
* **Adjacência:**
    * [cite_start]Dois **vértices** são adjacentes se são incidentes a uma mesma aresta[cite: 753].
    * [cite_start]Duas **arestas** são adjacentes se são incidentes a um mesmo vértice[cite: 758].
    * [cite_start]É uma relação entre elementos do mesmo tipo (vértice-vértice ou aresta-aresta)[cite: 774].

#### **1.4. Grau de um Vértice**
* [cite_start]**Definição:** O grau de um vértice $v$, denotado por $d(v)$, é o número de arestas incidentes a ele[cite: 797].
* [cite_start]**Vértice Isolado:** Um vértice com grau zero[cite: 839].
* [cite_start]**Vértice Folha (ou Terminal):** Um vértice com grau 1[cite: 840].
* [cite_start]**Propriedade 1 (Lema do Aperto de Mão):** Em qualquer grafo, a soma de todos os graus dos vértices é igual ao dobro do número de arestas: $\sum_{v \in V} d(v) = 2m$[cite: 892]. [cite_start]A prova se baseia no fato de que cada aresta contribui com +1 para o grau de exatamente dois vértices[cite: 916, 917].
    * **Aplicação:** (Lista 1, Q1, Q2) Para verificar se um grafo com uma dada sequência de graus pode existir, some os graus. Se o resultado for ímpar, o grafo é impossível, pois $2m$ é sempre par.
* [cite_start]**Propriedade 2 (Corolário):** O número de vértices de grau ímpar em qualquer grafo é sempre par[cite: 929].
    * **Aplicação:** (Lista 1, Q2, Q3) A questão da festa (Q3) é uma aplicação direta disso. Se o número de pessoas que conhecem um número ímpar de outras pessoas fosse ímpar, a soma dos graus seria ímpar, o que é impossível.

#### **1.5. Vizinhança e Conjuntos Independentes**
* [cite_start]**Vizinhança:** O conjunto de vizinhos de um vértice $x$, denotado $\Gamma(x)$, é o conjunto de todos os vértices adjacentes a $x$[cite: 1007, 1008].
* [cite_start]**Conjunto Independente:** Um conjunto de vértices é independente se nenhum par de seus elementos for adjacente[cite: 843].

### **Tópico 2: Classes de Grafos e Propriedades**

Fundamental para questões da **Lista 1**.

#### **2.1. Grafos Regulares**
* [cite_start]**Definição:** Um grafo é **k-regular** se todos os seus vértices possuem grau $k$[cite: 6746].
    * [cite_start]**Aplicação:** (Lista 1, Q8) Ciclos são 2-regulares[cite: 6757]. [cite_start]Grafos completos $K_n$ são (n-1)-regulares[cite: 3030, 6758].
* [cite_start]**Número de Arestas em Grafo k-regular:** Um grafo k-regular com n vértices possui $m = \frac{n \times k}{2}$ arestas[cite: 3034]. Isso deriva diretamente do Lema do Aperto de Mão.
    * **Aplicação:** (Lista 1, Q8b).

#### **2.2. Grafos Completos ($K_n$)**
* [cite_start]**Definição:** Um grafo simples é completo se existe uma aresta ligando cada par de vértices[cite: 1224, 6873]. [cite_start]É denotado por $K_n$[cite: 6874].
* **Número de Arestas:** $|E(K_n)| [cite_start]= \frac{n(n-1)}{2}$[cite: 3025, 992].
    * **Aplicação:** (Lista 1, Q5).

#### **2.3. Grafos Bipartidos**
* [cite_start]**Definição:** Um grafo é k-partido se seu conjunto de vértices V pode ser particionado em k subconjuntos disjuntos ($Y_1, ..., Y_k$) tal que não existam arestas entre vértices da mesma partição[cite: 6905]. [cite_start]Um grafo **bipartido** é um caso especial onde k=2[cite: 3032]. As partições são geralmente chamadas de X e Y.
* [cite_start]**Propriedade Chave:** Um grafo é bipartido se, e somente se, ele **não contém ciclos de comprimento ímpar**[cite: 4529]. A demonstração mostra que, em um ciclo, os vértices devem alternar entre as partições. [cite_start]Se o ciclo tem comprimento ímpar, a última aresta necessariamente ligará dois vértices da mesma partição, o que é uma contradição[cite: 4545, 4546, 4547, 4548].
    * **Aplicação:** (Lista 2, Q3).
* [cite_start]**Grafo Bipartido Completo ($K_{p,q}$):** Um grafo bipartido onde cada vértice de uma partição X (com p vértices) é adjacente a *todos* os vértices da partição Y (com q vértices)[cite: 6921].
* **Número de Arestas em $K_{p,q}$:** $|E(K_{p,q})| [cite_start]= p \times q$[cite: 3025].
    * **Aplicação:** (Lista 1, Q5).
* [cite_start]**Máximo de Arestas em Bipartido:** Se $G$ é um grafo bipartido com n vértices, então $m \le \frac{n^2}{4}$[cite: 3027].
    * **Aplicação:** (Lista 1, Q7).

#### **2.4. Grafo Complementar ($\overline{G}$)**
* [cite_start]**Definição:** Dado um grafo $G=(V, E)$, seu complementar $\overline{G}$ é um grafo com o mesmo conjunto de vértices $V$, mas que contém exatamente as arestas que **não** estão em $G$[cite: 6954].
    * **Aplicação:** (Lista 1, Q4, Q6).

### **Tópico 3: Subgrafos e Isomorfismo**

Conceitos importantes para a **Lista 1**.

#### **3.1. Subgrafos**
* [cite_start]**Definição:** $H$ é um subgrafo de $G$ ($H \subseteq G$) se $V(H) \subseteq V(G)$ e $E(H) \subseteq E(G)$[cite: 1104].
* [cite_start]**Subgrafo Gerador:** Um subgrafo $H$ tal que $V(H) = V(G)$[cite: 1150]. Ou seja, contém todos os vértices do grafo original.
* [cite_start]**Subgrafo Induzido por Vértices ($G[V']$):** Para um subconjunto de vértices $V' \subseteq V$, o subgrafo induzido contém os vértices de $V'$ e **todas** as arestas de $G$ que têm ambos os extremos em $V'$[cite: 1165, 1166].
* [cite_start]**Clique:** Um subgrafo que é completo[cite: 1225]. Uma clique é um subgrafo induzido por um conjunto de vértices mutuamente adjacentes.
    * **Aplicação:** (Lista 1, Q11g, Q13).

#### **3.2. Isomorfismo**
* [cite_start]**Definição:** Um isomorfismo entre dois grafos $G$ e $H$ é uma bijeção (uma correspondência um-para-um) $f: V(G) \to V(H)$ que preserva a adjacência[cite: 1791]. [cite_start]Ou seja, $\{u,v\} \in E(G)$ se e somente se $\{f(u), f(v)\} \in E(H)$[cite: 1792].
* **Condições Necessárias (Invariantes):** Para serem isomorfos, dois grafos **devem** ter:
    1.  [cite_start]O mesmo número de vértices ($|V(G)| = |V(H)|$)[cite: 3036].
    2.  [cite_start]O mesmo número de arestas ($|E(G)| = |E(H)|$)[cite: 3036].
    3.  [cite_start]A mesma sequência de graus (o grau de cada vértice é preservado)[cite: 3036].
* [cite_start]**Recíproca Falsa:** Ter as mesmas invariantes (n, m, sequência de graus) **não garante** que os grafos sejam isomorfos[cite: 3037]. É preciso encontrar um contra-exemplo: dois grafos com as mesmas propriedades, mas com estruturas de conexão diferentes.
    * **Aplicação:** (Lista 1, Q9, Q10).

### **Tópico 4: Percursos, Conexidade e Buscas**

Esses tópicos são o foco da **Lista 2**.

#### **4.1. Tipos de Percurso**
* [cite_start]**Percurso (Cadeia):** Uma sequência de arestas sucessivamente adjacentes[cite: 2018, 2019]. Vértices e arestas podem ser repetidos. [cite_start]Pode ser **aberto** ou **fechado**[cite: 2021, 2020].
* [cite_start]**Percurso Simples:** Não repete **arestas**[cite: 2221].
* [cite_start]**Percurso Elementar (Caminho):** Não repete **vértices** (e consequentemente, não repete arestas)[cite: 2235].
* [cite_start]**Relação:** Todo percurso elementar é simples, mas o contrário não é verdadeiro[cite: 4831, 2551]. Um percurso pode repetir vértices mas não arestas.
* [cite_start]**Ciclo:** Um percurso **simples** e **fechado**[cite: 2250].
* [cite_start]**Ciclo Elementar:** Um ciclo onde apenas o primeiro e o último vértice se repetem[cite: 2266].

#### **4.2. Conexidade em Grafos**
* [cite_start]**Grafo Conexo:** Existe um caminho entre quaisquer dois vértices do grafo[cite: 4057].
* [cite_start]**Componentes Conexas ($\omega(G)$):** Subgrafos conexos maximais de um grafo[cite: 4329]. [cite_start]Um grafo desconexo é formado por duas ou mais componentes conexas[cite: 4281].
* [cite_start]**Ponte:** Uma aresta `a` cuja remoção aumenta o número de componentes conexas do grafo, ou seja, $\omega(G-a) > \omega(G)$[cite: 2843, 4836]. [cite_start]Uma ponte não pode pertencer a nenhum ciclo[cite: 4835].
* [cite_start]**Teorema dos Vértices de Grau Ímpar:** Se um grafo tem **exatamente dois** vértices de grau ímpar, então existe um caminho que os liga[cite: 4515, 4516].
    * **Aplicação:** (Lista 2, Q4).

#### **4.3. Algoritmos de Busca (Caminhamento)**
* [cite_start]**Busca em Profundidade (DFS - Depth-First Search):** Explora o grafo "aprofundando-se" o máximo possível por um ramo antes de retroceder (backtracking)[cite: 7]. [cite_start]Utiliza uma estrutura de dados de pilha (implicitamente, na recursão)[cite: 133, 247].
    * [cite_start]**Aplicações:** Determinar componentes conexas de um grafo não orientado e componentes f-conexas de um digrafo[cite: 188, 4864, 4865].
* [cite_start]**Busca em Largura (BFS - Breadth-First Search):** Explora o grafo em "camadas", visitando todos os nós à mesma distância do nó inicial[cite: 216]. [cite_start]Utiliza uma estrutura de dados de fila[cite: 250, 269].
    * [cite_start]**Aplicações:** A árvore de visitas gerada por uma BFS a partir de um vértice `s` representa a **árvore de caminhos mínimos** (em número de arestas) de `s` para todos os outros vértices alcançáveis[cite: 4867].
    * **Aplicação:** (Lista 2, Q13).

### **Tópico 5: Digrafos (Grafos Orientados)**

Essencial para questões específicas da **Lista 2**.

#### **5.1. Conceitos de Digrafos**
* [cite_start]**Definição:** Um digrafo $G=(V, A)$ possui arcos, que são pares **ordenados** de vértices[cite: 3134, 3160, 7001].
* **Graus:**
    * [cite_start]**Grau de Entrada ($d^-(v)$):** Número de arcos que chegam em $v$[cite: 3212].
    * [cite_start]**Grau de Saída ($d^+(v)$):** Número de arcos que saem de $v$[cite: 3213].
* [cite_start]**Antecessores e Sucessores:** Se existe um arco $(u, v)$, $u$ é antecessor de $v$, e $v$ é sucessor de $u$[cite: 3211]. [cite_start]A vizinhança é dividida em conjunto de sucessores $\Gamma^+(u)$ e de antecessores $\Gamma^-(u)$[cite: 3259, 3256].
* **Fecho Transitivo:**
    * [cite_start]**Direto ($\hat{\Gamma}^+(u)$):** Conjunto de todos os vértices atingíveis a partir de $u$[cite: 3337].
    * [cite_start]**Inverso ($\hat{\Gamma}^-(u)$):** Conjunto de todos os vértices a partir dos quais $u$ é atingível[cite: 3433].
* [cite_start]**Grafo Subjacente:** O grafo não orientado obtido ao remover a direção de todos os arcos[cite: 3523]. [cite_start]Um digrafo pode ser acíclico, mas seu grafo subjacente pode ter ciclos[cite: 4841].

#### **5.2. Conexidade em Digrafos**
* [cite_start]**s-conexo (simplesmente):** O grafo subjacente é conexo[cite: 3591].
* [cite_start]**sf-conexo (semi-fortemente):** Para todo par de vértices u, v, existe um caminho de u para v **ou** de v para u[cite: 3608, 3609].
* [cite_start]**f-conexo (fortemente):** Para todo par de vértices u, v, existe um caminho de u para v **e** um caminho de v para u[cite: 3627, 3628].
* [cite_start]**Componentes f-conexas:** Subgrafos maximais f-conexos[cite: 3827].
    * **Aplicação:** (Lista 2, Q8).

### **Tópico 6: Caminhos Mínimos e Distâncias**

Material principal para a **Lista 3**.

#### **6.1. Medidas de Distância**
* [cite_start]**Distância $d(v, w)$:** O comprimento (número de arestas ou soma dos pesos) do caminho mais curto entre $v$ e $w$[cite: 5746, 4780].
* [cite_start]**Excentricidade $E(v)$:** A maior distância de $v$ para qualquer outro vértice do grafo[cite: 5771]. [cite_start]$E(v) = \max_{w \in V} d(v, w)$[cite: 5772].
* [cite_start]**Raio:** A **menor** das excentricidades dos vértices[cite: 5784, 4784].
* [cite_start]**Diâmetro:** A **maior** das excentricidades dos vértices[cite: 5811, 4786].
* [cite_start]**Centro:** O conjunto de vértices cuja excentricidade é igual ao raio (excentricidade mínima)[cite: 5798, 4785].
* [cite_start]**Vértice Periférico:** Um vértice cuja excentricidade é igual ao diâmetro[cite: 5812, 4787].
    * **Aplicação:** (Lista 3, Q3).

#### **6.2. Algoritmos de Caminho Mínimo**
* [cite_start]**Condição de Existência:** Para que o problema de caminho mínimo seja bem definido, o grafo não pode conter **ciclos de comprimento negativo**[cite: 4960]. [cite_start]Se houver, pode-se percorrer o ciclo infinitamente para obter um "caminho" de custo $-\infty$[cite: 4954].
    * **Aplicação:** (Lista 3, Q5).
* **Algoritmo de Dijkstra:**
    * [cite_start]Resolve o problema de caminho mínimo a partir de **uma fonte** para todos os outros vértices[cite: 4974, 5006].
    * [cite_start]**Restrição:** Funciona apenas para grafos com pesos de aresta/arco **não-negativos** ($c_{ij} \ge 0$)[cite: 4976]. [cite_start]Se houver arestas negativas, o algoritmo pode falhar em encontrar o caminho correto, pois sua premissa "gulosa" de que uma vez que um vértice é "fechado" sua distância é final não se sustenta mais[cite: 5364].
    * **Aplicação:** (Lista 3, Q4, Q6).
* **Algoritmo de Bellman-Ford:**
    * [cite_start]Também resolve o problema de caminho mínimo a partir de uma fonte[cite: 5972].
    * **Vantagem:** Funciona mesmo com pesos de aresta **negativos**.
    * [cite_start]**Detecção de Ciclo Negativo:** Se existir um ciclo negativo, o algoritmo entra em loop, atualizando os custos indefinidamente[cite: 6344]. [cite_start]Se for possível fazer uma atualização na $n$-ésima iteração (após as $n-1$ iterações padrão), isso indica a presença de um ciclo negativo[cite: 6345].
* **Algoritmo de Floyd-Warshall:**
    * [cite_start]Resolve o problema de caminho mínimo **entre todos os pares** de vértices[cite: 6369, 6370].
    * [cite_start]A ideia é permitir, a cada iteração $k$, que o vértice $k$ seja usado como um vértice intermediário nos caminhos[cite: 6375, 6376]. [cite_start]A fórmula de recorrência é: $A^{k+1}(i,j) = \min\{A^{k}(i,j), A^{k}(i,k+1) + A^{k}(k+1,j)\}$[cite: 6536].

### **Tópico 7: Percursos Especiais: Euleriano e Hamiltoniano**

Tópicos abordados na **Lista 3** e na **Lista 1**.

#### **7.1. Grafos Eulerianos**
* [cite_start]**Ciclo Euleriano (Linha de Euler):** Um ciclo que passa por **cada aresta** do grafo exatamente uma vez[cite: 2659].
* [cite_start]**Grafo Euleriano:** Um grafo que possui um ciclo euleriano[cite: 2675].
* [cite_start]**Teorema de Euler:** Um grafo conexo é euleriano se, e somente se, **todos os seus vértices têm grau par**[cite: 2748].
* [cite_start]**Caminho Euleriano:** Um caminho (não necessariamente fechado) que passa por cada aresta exatamente uma vez[cite: 2676].
* **Condição para Caminho Euleriano:** Um grafo conexo tem um caminho euleriano se, e somente se, ele tem **zero ou dois** vértices de grau ímpar. Se tiver dois, o caminho começa em um deles e termina no outro.
    * **Aplicação:** (Lista 1, Q14 - Dominós) Modelar as peças como um grafo onde os números {1,2,3,4,5} são vértices e as peças de dominó são arestas. O objetivo é encontrar um caminho euleriano. (Lista 3, Q7).

#### **7.2. Grafos Hamiltonianos**
* [cite_start]**Ciclo Hamiltoniano:** Um ciclo que passa por **cada vértice** do grafo exatamente uma vez[cite: 2931].
* [cite_start]**Grafo Hamiltoniano:** Um grafo que possui um ciclo hamiltoniano[cite: 2932].
* [cite_start]**Caminho Hamiltoniano:** Um caminho que passa por cada vértice exatamente uma vez[cite: 2934].
* **Propriedades:**
    * Diferente dos grafos eulerianos, **não há uma condição simples e necessária/suficiente** para determinar se um grafo é hamiltoniano.
    * [cite_start]Um grafo hamiltoniano não precisa ser euleriano, e vice-versa[cite: 2991, 2992, 2993, 2994].
    * **Hamiltoniano e Bipartido:** Se um grafo bipartido $G=(A \cup B, E)$ for hamiltoniano, as partições devem ter o mesmo tamanho, $|A| [cite_start]= |B|$, pois o ciclo deve alternar entre os vértices de A e B. Se um grafo bipartido tiver um número ímpar de vértices, ele não pode ser hamiltoniano[cite: 4766]. Se um caminho hamiltoniano existir em um grafo bipartido, o tamanho das partições pode diferir em no máximo 1: $| |A| - |B| | [cite_start]\le 1$[cite: 4769].
    * **Aplicação:** (Lista 3, Q1, Q2, Q7).
