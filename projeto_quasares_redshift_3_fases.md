# Identificação automática de quasares e estimativa de redshift fotométrico usando dados multibanda de surveys astronômicos

Sim — esse é um bom enunciado para o tema do projeto, e ele está alinhado com a formulação já usada no mini plano anterior: classificação fotométrica de objetos astronômicos e estimativa de redshift fotométrico de quasares com dados multibanda de surveys astronômicos.[file:349][file:309]

De forma mais objetiva, o foco central do projeto pode ser descrito assim: desenvolver e avaliar abordagens computacionais para identificar quasares automaticamente e estimar seu redshift fotométrico a partir de dados observacionais provenientes de surveys astronômicos multibanda.[file:349]

## Problema do projeto

Surveys fotométricos modernos produzem grandes volumes de dados e tornam inviável a inspeção manual de todos os objetos observados.[file:349] Nesse contexto, métodos de machine learning ajudam a separar classes astronômicas, selecionar candidatos a quasares e explorar relações entre magnitudes, cores fotométricas e distância cosmológica aproximada via redshift fotométrico.[file:349][file:309]

O problema científico-computacional do projeto é investigar como diferentes representações dos dados astronômicos — imagens, atributos tabulares e posteriormente a combinação entre ambos — podem contribuir para a identificação automática de quasares e para a estimativa de redshift fotométrico.[file:309][file:349]

## Estrutura do experimento

O experimento será desenvolvido em **três partes**, de forma incremental e comparável, para que cada etapa produza resultados próprios e ao mesmo tempo prepare a etapa seguinte.[file:309][file:349]

### Parte 1 — Benchmark de modelos baseados em imagem

A primeira fase será dedicada apenas a modelos que trabalham com imagens astronômicas, porque essa etapa inicial está mais alinhada à avaliação atual focada em processamento digital de imagens.[file:309] Serão utilizados recortes de imagem associados aos objetos astronômicos, com o objetivo de classificar inicialmente classes como estrela, galáxia e quasar e medir quais arquiteturas apresentam melhor desempenho nessa representação.[file:309][file:349]

Nessa fase, o projeto funcionará como um benchmark de modelos de classificação por imagem. A ideia é comparar arquiteturas visuais apropriadas para classificação e identificar quais delas oferecem melhor equilíbrio entre desempenho, custo computacional e interpretabilidade experimental antes de qualquer integração com dados tabulares.[file:309]

### Parte 2 — Benchmark de modelos baseados em dados tabulares

A segunda fase será dedicada aos modelos que utilizam atributos fotométricos tabulares, como magnitudes, cores e outras variáveis derivadas dos surveys astronômicos.[file:349][file:309] Essa etapa permitirá construir um benchmark separado para dados estruturados, com foco especial na classificação fotométrica de objetos astronômicos e, em seguida, na estimativa de redshift fotométrico para quasares.[file:349]

Nessa fase, os experimentos serão organizados de forma supervisionada, com separação entre treino, validação e teste, evitando vazamento de informação e permitindo medir o desempenho de algoritmos clássicos e robustos para dados tabulares.[file:349] O objetivo é entender até que ponto os atributos fotométricos sozinhos já são suficientes para identificar quasares e prever redshift de maneira útil.[file:349]

### Parte 3 — Comparação final e abordagem híbrida

Depois de concluídas as duas trilhas anteriores, a terceira fase fará a comparação entre os melhores modelos de imagem e os melhores modelos tabulares.[file:309][file:349] Essa comparação servirá para responder qual representação dos dados é mais vantajosa para cada subtarefa: identificação de quasares, classificação mais ampla de objetos astronômicos e estimativa de redshift fotométrico.[file:309][file:349]

Em seguida, será investigada uma abordagem híbrida, combinando informações vindas das imagens e dos atributos tabulares em um mesmo experimento multimodal.[file:309] Essa etapa será tratada como evolução natural do projeto e deverá aproveitar os melhores resultados das fases anteriores, em vez de tentar construir um modelo multimodal desde o início.[file:309]

## Lógica de desenvolvimento

A ordem em três fases foi escolhida para manter o projeto viável, reprodutível e cientificamente organizado.[file:309][file:349] Primeiro, obtém-se um benchmark sólido em visão computacional; depois, um benchmark equivalente em dados tabulares; por fim, compara-se as duas abordagens e explora-se uma possível fusão entre elas.[file:309]

Essa estratégia reduz complexidade inicial, facilita a documentação dos experimentos e produz marcos claros de entrega.[file:309][file:349] Além disso, ela mantém o tema central do projeto — identificação automática de quasares e estimativa de redshift fotométrico com dados multibanda — enquanto distribui a implementação em etapas mais controláveis.[file:349][file:309]

## Formulação consolidada do tema

Se quiser uma formulação curta e oficial para usar no repositório, no README ou em documentos do projeto, uma boa versão é esta:

**Tema do projeto:** Identificação automática de quasares e estimativa de redshift fotométrico usando dados multibanda de surveys astronômicos.[file:349]

**Descrição resumida:** O projeto investiga, em três fases, o uso de modelos de classificação por imagem, modelos de aprendizado sobre dados tabulares e abordagens híbridas para identificar quasares automaticamente e estimar redshift fotométrico a partir de dados de surveys astronômicos multibanda.[file:309][file:349]
