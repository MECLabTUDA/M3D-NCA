
import numpy as np

from skimage import measure
import plotly.graph_objects as go


def named_mesh_from_volume(data: np.ndarray, name: str, opacity: float, colorscale: str):
    """Creates a named plotly mesh from data array.
    
    Args:
        data: 3d data, shape has to meet requirements of skimage.measure.marching_cubes
        name: name for plotly object
        opacity: opacity passed to plotly
        colorscale: passed to plotly
    """
    verts, faces, _, _ = measure.marching_cubes(data, allow_degenerate=True)
    x, y, z = zip(*verts)  
    I, J, K = faces.T
    tri = go.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        opacity=opacity,
        name=name,
        colorscale=colorscale
    )
    return tri