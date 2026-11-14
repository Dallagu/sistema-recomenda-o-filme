# Sistema de recomendação de filmes

Protótipo de recomendação baseada em conteúdo que compara as sinopses do arquivo `movies_metadata.csv` usando TF-IDF e similaridade do cosseno.

## Origem do protótipo

A versão inicial do código, do notebook e de seus textos foi produzida com participação substancial de IA em uma sessão de assistência. O uso acadêmico desse material precisa respeitar as regras da atividade e refletir a contribuição efetiva do estudante.

## Arquivos

- `Sistema_recomendacao_de_filmes.ipynb` reúne a explicação, o código e as saídas salvas da execução.
- `recomendador.py` permite consultar títulos e executar a avaliação pelo terminal.
- `requirements.txt` registra a versão de NumPy usada no experimento.

## Base de dados

Use o arquivo `movies_metadata.csv` de [The Movies Dataset, publicado por Rounak Banik](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset). Coloque o CSV na mesma pasta do código ou informe seu caminho no comando de execução.

O arquivo original utilizado tem 45.466 registros e 24 colunas. Seu MD5 é `42bf3ef8c208a01a4776955875978b1e`.

## Execução no Google Colab

Abra o notebook no Colab e execute as células de cima para baixo. Envie `movies_metadata.csv` quando o notebook solicitar. Altere a variável `FILME` para consultar outro título exato ou ID do TMDB.

## Execução local

O experimento utilizou Python 3.12.14 e NumPy 2.3.5. Use Python 3.12 para reproduzir esse ambiente.

```sh
python -m pip install -r requirements.txt
python recomendador.py movies_metadata.csv --filme "Toy Story"
python recomendador.py movies_metadata.csv --avaliar
```

## Exemplo de entrada e saída

Entrada do teste com o título `Toy Story`, ID do TMDB `862`.

```text
1. Toy Story 3 (2010) | ID 10193 | 0.4991
2. Toy Story 2 (1999) | ID 863   | 0.4683
3. The 40 Year Old Virgin (2005) | ID 6957 | 0.2854
```

As três linhas representam as primeiras posições do ranking de até dez filmes. A terceira sugestão mostra uma limitação da comparação lexical. O nome Andy, usado para personagens diferentes, aproximou as sinopses.

Para a entrada `Finding Nemo`, ID `12`, o primeiro resultado foi `Finding Dory`, ID `127380`, com similaridade `0.2850`.

## Avaliação

O catálogo tratado tem 44.317 filmes e o vocabulário contém 37.419 termos. O teste usa como relevantes os outros filmes da mesma coleção e considera 4.030 consultas de 1.287 coleções com pelo menos dois títulos válidos.

| Métrica | TF-IDF e cosseno | Mais votados |
|---|---:|---:|
| Hit@10 | 61,76% | 0,20% |
| Precisão@10 | 13,71% | 0,02% |
| Recall@10 | 43,72% | 0,12% |
| MRR@10 | 0,46467 | 0,00050 |

O Hit@10 de 61,76% corresponde a 2.489 consultas com pelo menos um integrante da coleção entre os dez resultados. As métricas avaliam recuperação dentro de um catálogo fixo. A coleção atua como referência aproximada de relevância, e sugestões pertinentes de outras coleções ficam fora dessa contagem. Avaliar satisfação e preferências pessoais depende de evidências adicionais.

## Referências técnicas

- [TF-IDF na documentação do scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting)
- [Similaridade do cosseno](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
- [MovieLens, para possíveis experimentos com avaliações individuais](https://grouplens.org/datasets/movielens/)
