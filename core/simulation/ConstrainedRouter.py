from __future__ import annotations

import pickle
import argparse
import networkx as nx


class ConstrainedRouter :

	SOURCE : str = "SRC"
	TARGET : str = "DST"

	def __init__(self, digraph : nx.DiGraph, oriented : dict) :
		self.digraph : nx.DiGraph = digraph
		self.oriented : dict = oriented
		self.cache : dict = {}
		self._from : dict = {}
		self._to : dict = {}
		for (a, b), data in oriented.items() :
			self._from.setdefault(a, []).append(((a, b), data["length"]))
			self._to.setdefault(b, []).append((a, b))

	@classmethod
	def fromFiles(cls, digraph_path : str, oriented_path : str) -> "ConstrainedRouter" :
		with open(digraph_path, "rb") as f :
			digraph = pickle.load(f)
		with open(oriented_path, "rb") as f :
			oriented = pickle.load(f)
		return cls(digraph, oriented)

	def shortestPath(self, source : int, target : int) -> list[int] | None :
		key = (source, target)
		if key in self.cache :
			return self.cache[key]

		# Terminals are attached in place then removed : copying the digraph costs
		# more than the search itself.
		graph = self.digraph
		spawned = set()
		for edge, length in self._from.get(source, ()) :
			if edge not in graph :
				spawned.add(edge)
			graph.add_edge(self.SOURCE, edge, weight=length)
		for edge in self._to.get(target, ()) :
			if edge not in graph :
				spawned.add(edge)
			graph.add_edge(edge, self.TARGET, weight=0)
		try :
			node_path = nx.shortest_path(graph, self.SOURCE, self.TARGET, weight="weight")
			edges = [n for n in node_path if n not in (self.SOURCE, self.TARGET)]
			path = [edges[0][0]] + [e[1] for e in edges]
		except (nx.NetworkXNoPath, nx.NodeNotFound) :
			path = None
		finally :
			for node in (self.SOURCE, self.TARGET, *spawned) :
				if node in graph :
					graph.remove_node(node)

		self.cache[key] = path
		return path

	def pathLength(self, switches : list[int]) -> float :
		return sum(self.oriented[(a, b)]["length"] for a, b in zip(switches, switches[1:]))


if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description="Compute a turn-constrained shortest path")
	parser.add_argument("-d", type=str, required=True, help="Path to the constrained digraph pickle")
	parser.add_argument("-e", type=str, required=True, help="Path to the oriented edges pickle")
	parser.add_argument("-s", type=int, required=True, help="Source switch id")
	parser.add_argument("-t", type=int, required=True, help="Target switch id")
	args = parser.parse_args()
	router = ConstrainedRouter.fromFiles(args.d, args.e)
	path = router.shortestPath(args.s, args.t)
	if path is None :
		print(f"No valid path from {args.s} to {args.t}.")
	else :
		print(f"Path ({router.pathLength(path):.1f} m) : {path}")
