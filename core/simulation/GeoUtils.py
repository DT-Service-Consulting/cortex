import math


class GeoUtils :

	@staticmethod
	def bearing(p1 : list[float], p2 : list[float]) -> float :
		lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
		lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
		dlon = lon2 - lon1
		x = math.sin(dlon) * math.cos(lat2)
		y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
		return math.degrees(math.atan2(x, y)) % 360

	@staticmethod
	def turnDeviation(in_bearing : float, out_bearing : float) -> float :
		d = abs(out_bearing - in_bearing) % 360
		return min(d, 360 - d)

	@staticmethod
	def firstDistinct(points : list[list[float]], reverse : bool = False) -> tuple[list[float], list[float]] :
		seq = list(reversed(points)) if reverse else points
		ref = seq[0]
		for p in seq[1:] :
			if (p[0], p[1]) != (ref[0], ref[1]) :
				return ref, p
		return ref, seq[-1]
