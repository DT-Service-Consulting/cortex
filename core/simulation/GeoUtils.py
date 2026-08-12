import math


class GeoUtils :

	R = 6371000.0

	@staticmethod
	def haversine(p1 : list[float], p2 : list[float]) -> float :
		lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
		lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
		dlat = lat2 - lat1
		dlon = lon2 - lon1
		a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
		return 2 * GeoUtils.R * math.asin(min(1.0, math.sqrt(a)))

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

	@staticmethod
	def _toLocal(origin : list[float], p : list[float]) -> tuple[float, float] :
		lat0 = math.radians(origin[0])
		x = math.radians(p[1] - origin[1]) * GeoUtils.R * math.cos(lat0)
		y = math.radians(p[0] - origin[0]) * GeoUtils.R
		return x, y

	@staticmethod
	def pointAlongPath(path : list[list[float]], dist_m : float) -> list[float] :
		if not path :
			raise ValueError("empty path")
		if dist_m <= 0 :
			return list(path[0][:2])
		acc = 0.0
		for a, b in zip(path, path[1:]) :
			seg = GeoUtils.haversine(a, b)
			if seg <= 0 :
				continue
			if acc + seg >= dist_m :
				t = (dist_m - acc) / seg
				return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
			acc += seg
		return list(path[-1][:2])

	@staticmethod
	def projectOnPath(path : list[list[float]], point : list[float]) -> float :
		if len(path) < 2 :
			return 0.0
		best_d2 = float("inf")
		best_along = 0.0
		acc = 0.0
		for a, b in zip(path, path[1:]) :
			ax, ay = GeoUtils._toLocal(a, a)
			bx, by = GeoUtils._toLocal(a, b)
			px, py = GeoUtils._toLocal(a, point)
			dx, dy = bx - ax, by - ay
			seg2 = dx * dx + dy * dy
			if seg2 <= 0 :
				continue
			t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
			qx, qy = ax + t * dx, ay + t * dy
			d2 = (px - qx) ** 2 + (py - qy) ** 2
			seg_len = math.sqrt(seg2)
			if d2 < best_d2 :
				best_d2 = d2
				best_along = acc + t * seg_len
			acc += seg_len
		return best_along
