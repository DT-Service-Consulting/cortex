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

	@classmethod
	def fromFiles(cls, digraph_path : str, oriented_path : str) -> "ConstrainedRouter" :
		with open(digraph_path, "rb") as f :
			digraph = pickle.load(f)
		with open(oriented_path, "rb") as f :
			oriented = pickle.load(f)
		return cls(digraph, oriented)

	def shortestPath(self, source : int, target : int) -> list[int] | None :
		graph = self.digraph.copy()
		for (a, b) in self.oriented :
			if a == source :
				graph.add_edge(self.SOURCE, (a, b), weight=self.oriented[(a, b)]["length"])
			if b == target :
				graph.add_edge((a, b), self.TARGET, weight=0)
		try :
			node_path = nx.shortest_path(graph, self.SOURCE, self.TARGET, weight="weight")
		except (nx.NetworkXNoPath, nx.NodeNotFound) :
			return None
		edges = [n for n in node_path if n not in (self.SOURCE, self.TARGET)]
		return [edges[0][0]] + [e[1] for e in edges]

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
