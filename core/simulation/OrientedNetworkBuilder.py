import ast
import pickle
import argparse
import pandas as pd
import networkx as nx
try:
	from .GeoUtils import GeoUtils
except ImportError:
	from GeoUtils import GeoUtils


class OrientedNetworkBuilder :

	def __init__(self, filename : str, separator : str, output_dir : str, max_turn_deg : float = 60.0) :
		self.filename : str = filename
		self.separator : str = separator
		self.output_dir : str = output_dir
		self.max_turn_deg : float = max_turn_deg
		self.output_filenames : dict[str, str] = {
			"oriented" : f"{self.output_dir}/oriented_edges.pkl",
			"digraph" : f"{self.output_dir}/constrained_digraph.pkl",
			"constraints" : f"{self.output_dir}/forbidden_transitions.csv",
		}
		self.dataframe : pd.DataFrame = None
		self.oriented : dict = dict()
		self.incoming : dict = dict()
		self.outgoing : dict = dict()
		self.digraph : nx.DiGraph = nx.DiGraph()
		self.forbidden : list[dict] = list()

	def extractData(self) -> None :
		print("Extracting data...")
		self.dataframe = pd.read_csv(self.filename, sep=self.separator)
		print("Data extraction completed successfully.")

	def buildOrientedEdges(self) -> None :
		print("Building oriented edges...")
		for _, row in self.dataframe.iterrows() :
			path = ast.literal_eval(row["Path"])
			if len(path) < 2 :
				continue
			a, b, length = row["Departure_switch"], row["Arrival_switch"], row["Length_m"]
			self.oriented[(a, b)] = {"path" : path, "length" : length}
			self.oriented[(b, a)] = {"path" : path[::-1], "length" : length}
		print(f"Oriented edges built : {len(self.oriented)}")

	def computeBearings(self) -> None :
		print("Computing bearings...")
		for (a, b), edge in self.oriented.items() :
			p_last, p_prev = GeoUtils.firstDistinct(edge["path"], reverse=True)
			p_first, p_next = GeoUtils.firstDistinct(edge["path"], reverse=False)
			edge["in_bearing"] = GeoUtils.bearing(p_prev, p_last)
			edge["out_bearing"] = GeoUtils.bearing(p_first, p_next)
			self.incoming.setdefault(b, []).append((a, b))
			self.outgoing.setdefault(a, []).append((a, b))
		print("Bearings computed successfully.")

	def buildConstrainedGraph(self) -> None :
		print("Building constrained digraph...")
		for switch in self.incoming :
			for (a, b) in self.incoming[switch] :
				in_bearing = self.oriented[(a, b)]["in_bearing"]
				for (b2, c) in self.outgoing.get(switch, []) :
					if c == a :
						continue
					deviation = GeoUtils.turnDeviation(in_bearing, self.oriented[(b2, c)]["out_bearing"])
					if deviation <= self.max_turn_deg :
						self.digraph.add_edge((a, b), (b2, c), weight=self.oriented[(b2, c)]["length"])
					else :
						self.forbidden.append({
							"from_switch" : a,
							"via_switch" : switch,
							"to_switch" : c,
							"deviation_deg" : round(deviation, 2),
						})
		print(f"Allowed transitions : {self.digraph.number_of_edges()} | Forbidden transitions : {len(self.forbidden)}")

	def loadData(self) -> None :
		print("Storing artifacts...")
		with open(self.output_filenames["oriented"], "wb") as f :
			pickle.dump(self.oriented, f)
		with open(self.output_filenames["digraph"], "wb") as f :
			pickle.dump(self.digraph, f)
		pd.DataFrame(self.forbidden).to_csv(self.output_filenames["constraints"], index=False)
		print("Artifacts stored successfully.")

	def run(self) -> None :
		print("Starting oriented network construction.")
		self.extractData()
		self.buildOrientedEdges()
		self.computeBearings()
		self.buildConstrainedGraph()
		self.loadData()
		print("Oriented network construction completed successfully.")


if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description="Build a turn-constrained oriented railway network")
	parser.add_argument("-i", type=str, required=True, help="Path to the main tracks CSV file")
	parser.add_argument("-o", type=str, default=".", help="Path to the output directory")
	parser.add_argument("-sep", type=str, default=";", help="CSV separator (default: ';')")
	parser.add_argument("-max_turn", type=float, default=60.0, help="Maximum allowed turn deviation in degrees")
	args = parser.parse_args()
	builder = OrientedNetworkBuilder(args.i, args.sep, args.o, args.max_turn)
	builder.run()
