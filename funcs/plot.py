


def drawMesh(fem, nodes, val, title=None, fileName=None, ifsave=False):
	import matplotlib
	import matplotlib.pyplot as plt
	import matplotlib.collections
	import numpy as np

	nodesX = []
	nodesY = []
	elements = []
	stresses = []

	num_meshes = fem.shape[0]

	nodeIdx = 0
	for idx in range(num_meshes):
		center = fem[idx, :2]
		vertices = nodes[idx].tolist()
		for vertex in vertices:
			nodesX.append(center[0] + vertex[0])
			nodesY.append(center[1] + vertex[1])
		elements.append([nodeIdx, nodeIdx + 1, nodeIdx + 2])
		nodeIdx += 3
		stresses.append(val[idx])

	def quatplot(y, z, quatrangles, values, ax=None, **kwargs):
    # refer to
		# https://stackoverflow.com/questions/52202014/how-can-i-plot-2d-fem-results-using-matplotlib
		if not ax: ax = plt.gca()
		yz = np.c_[y, z]
		verts = yz[quatrangles]
		pc = matplotlib.collections.PolyCollection(verts, **kwargs)
		pc.set_array(values)
		ax.add_collection(pc)
		ax.autoscale()
		return pc

	nodesX = np.array(nodesX)
	nodesY = np.array(nodesY)
	elements = np.array(elements)
	stresses = np.array(stresses)

	fig, ax = plt.subplots(dpi=300)
	ax.set_aspect('equal')

	# pc = quatplot(y,z, np.asarray(elements), values, ax=ax,
	#          edgecolor="crimson", cmap="rainbow")
	pc = quatplot(nodesX, nodesY, np.asarray(elements), stresses, ax=ax,
				  cmap="rainbow")
	fig.colorbar(pc, ax=ax)

	if not title is None:
		ax.set_title(title)

	if ifsave ==True:
		tmpPath ="./" + fileName + "_" + title + ".jpg"
		plt.savefig(tmpPath)

	plt.show()
	fig.clear()
	plt.close(fig)
