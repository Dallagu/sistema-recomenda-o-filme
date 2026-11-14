"""Recomenda filmes pelo conteúdo de suas sinopses. Requer NumPy."""

import argparse
import ast
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from math import log, sqrt
import re

import numpy as np


TOKEN = re.compile(r"[^\W\d_]{2,}")
SINOPSES_AUSENTES = {
    "no movie overview available",
    "no movie overview available, please add one at themoviedb.org",
    "no overview",
    "no overview found",
    "no overview yet",
    "no plot overview available",
    "not available",
}
K = 10


@dataclass
class Filme:
    id: int
    titulo: str
    ano: str
    sinopse: str
    votos: int
    colecao: int | None


def carregar_catalogo(caminho):
    filmes = {}
    contagens = Counter()
    with open(caminho, encoding="utf-8-sig", newline="") as arquivo:
        for linha in csv.DictReader(arquivo):
            contagens["linhas lidas"] += 1
            identificador = (linha["id"] or "").strip()
            if not re.fullmatch(r"[1-9][0-9]*", identificador):
                contagens["ID inválido"] += 1
                continue
            titulo = (linha["title"] or "").strip()
            if not titulo:
                contagens["sem título"] += 1
                continue
            sinopse = " ".join((linha["overview"] or "").split())
            if not TOKEN.search(sinopse):
                contagens["sem sinopse utilizável"] += 1
                continue
            if sinopse.casefold().rstrip(".") in SINOPSES_AUSENTES:
                contagens["marcador de sinopse ausente"] += 1
                continue

            id_filme = int(identificador)
            anterior = filmes.get(id_filme)
            if anterior is not None:
                contagens["linhas repetidas por ID"] += 1
                if len(anterior.sinopse) >= len(sinopse):
                    continue

            votos = (linha["vote_count"] or "").strip()
            colecao = (linha["belongs_to_collection"] or "").strip()
            filmes[id_filme] = Filme(
                id_filme,
                titulo,
                (linha["release_date"] or "")[:4],
                sinopse,
                int(votos) if votos.isdecimal() else 0,
                ast.literal_eval(colecao)["id"] if colecao else None,
            )
    return sorted(filmes.values(), key=lambda filme: filme.id), contagens


class IndiceTfidf:
    def __init__(self, filmes):
        documentos = [Counter(TOKEN.findall(filme.sinopse.lower())) for filme in filmes]
        frequencia = Counter(termo for doc in documentos for termo in doc)
        total = len(filmes)
        idf = {
            termo: log((1 + total) / (1 + ocorrencias)) + 1
            for termo, ocorrencias in frequencia.items()
            if 2 <= ocorrencias <= 0.20 * total
        }
        self.vetores = []
        ocorrencias_por_termo = defaultdict(list)
        for posicao, doc in enumerate(documentos):
            pesos = {termo: qtd * idf[termo] for termo, qtd in doc.items() if termo in idf}
            norma = sqrt(sum(peso ** 2 for peso in pesos.values()))
            vetor = {termo: peso / norma for termo, peso in pesos.items()}
            self.vetores.append(vetor)
            for termo, peso in vetor.items():
                ocorrencias_por_termo[termo].append((posicao, peso))

        self.ocorrencias = {}
        for termo, lista in ocorrencias_por_termo.items():
            posicoes, pesos = zip(*lista)
            self.ocorrencias[termo] = (np.asarray(posicoes, dtype=np.intp), np.asarray(pesos))

    def recomendar(self, consulta, k=K):
        similaridades = np.zeros(len(self.vetores))
        for termo, peso in self.vetores[consulta].items():
            posicoes, pesos = self.ocorrencias[termo]
            similaridades[posicoes] += peso * pesos
        similaridades[consulta] = 0

        candidatos = np.flatnonzero(similaridades > 0)
        ordem = np.lexsort((candidatos, -similaridades[candidatos]))[:k]
        return [(int(i), float(similaridades[i])) for i in candidatos[ordem]]


def avaliar(filmes, indice):
    colecoes = defaultdict(set)
    for posicao, filme in enumerate(filmes):
        if filme.colecao is not None:
            colecoes[filme.colecao].add(posicao)

    popularidade = sorted(range(len(filmes)), key=lambda i: (-filmes[i].votos, filmes[i].id))
    somas = {"TF-IDF + cosseno": Counter(), "Mais votados": Counter()}
    consultas = 0
    colecoes_avaliadas = 0
    for grupo in colecoes.values():
        if len(grupo) < 2:
            continue
        colecoes_avaliadas += 1
        for consulta in sorted(grupo):
            consultas += 1
            relevantes = grupo - {consulta}
            ranking = {
                "TF-IDF + cosseno": [i for i, _ in indice.recomendar(consulta)],
                "Mais votados": [i for i in popularidade[: K + 1] if i != consulta][:K],
            }
            for metodo, lista in ranking.items():
                posicoes = [pos for pos, i in enumerate(lista, start=1) if i in relevantes]
                somas[metodo]["hit"] += bool(posicoes)
                somas[metodo]["precisao"] += len(posicoes) / K
                somas[metodo]["recall"] += len(posicoes) / len(relevantes)
                somas[metodo]["mrr"] += 1 / posicoes[0] if posicoes else 0

    print(f"\nAvaliação top {K} | {consultas} consultas de {colecoes_avaliadas} coleções")
    if not consultas:
        return
    for metodo, medidas in somas.items():
        print(
            f"{metodo}: {medidas['hit']} acertos por consulta; "
            f"Hit@{K}={medidas['hit'] / consultas:.4%}; "
            f"Precisão@{K}={medidas['precisao'] / consultas:.4%}; "
            f"Recall@{K}={medidas['recall'] / consultas:.4%}; "
            f"MRR@{K}={medidas['mrr'] / consultas:.5f}"
        )


def mostrar_recomendacoes(filmes, indice, consulta):
    filme = filmes[consulta]
    print(f"\n{filme.titulo} ({filme.ano or 'ano desconhecido'}) | ID {filme.id}")
    recomendacoes = indice.recomendar(consulta)
    if not recomendacoes:
        print("Sem filmes com similaridade positiva.")
    for posicao, (i, escore) in enumerate(recomendacoes, start=1):
        filme = filmes[i]
        print(f"{posicao:2}. {filme.titulo} ({filme.ano or 'ano desconhecido'}) | ID {filme.id} | {escore:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="movies_metadata.csv", help="caminho do CSV")
    parser.add_argument("--filme", help="ID do TMDB ou título exato no catálogo")
    parser.add_argument("--avaliar", action="store_true", help="calcula as métricas do relatório")
    args = parser.parse_args()
    filmes, contagens = carregar_catalogo(args.csv)
    if not filmes:
        parser.error("o catálogo não contém filmes com sinopse utilizável")

    if args.filme:
        termo = args.filme.strip().casefold()
        consultas = [i for i, f in enumerate(filmes) if str(f.id) == termo]
        if not consultas:
            consultas = [i for i, f in enumerate(filmes) if f.titulo.casefold() == termo]
        if len(consultas) > 1:
            opcoes = ", ".join(f"{filmes[i].titulo} ({filmes[i].ano}) ID {filmes[i].id}" for i in consultas)
            parser.error(f"título ambíguo. Consulte pelo ID: {opcoes}")
        if not consultas:
            parser.error(f"filme não encontrado: {args.filme}")
    else:
        consultas = [i for i, f in enumerate(filmes) if f.id in {862, 603, 12}]

    print("Limpeza:", dict(contagens))
    print(f"Catálogo: {len(filmes)} filmes")
    indice = IndiceTfidf(filmes)
    print(f"Vocabulário: {len(indice.ocorrencias)} termos")
    print(f"Filmes sem termos no vocabulário: {sum(not vetor for vetor in indice.vetores)}")
    for consulta in consultas:
        mostrar_recomendacoes(filmes, indice, consulta)
    if args.avaliar:
        avaliar(filmes, indice)


if __name__ == "__main__":
    main()
