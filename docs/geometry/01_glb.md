# Your first GLB

From the TrueFace3D root, run these commands in PowerShell:

```powershell
npm --prefix ui ci
python -m http.server 8765 --bind 127.0.0.1 --directory ui
```

Open http://127.0.0.1:8765/ in Edge or Chrome. Stop the server with Ctrl+C.

1. Move **Top vertex X** to 0.65. You moved one of the triangle's three points along the horizontal axis.
2. Orbit around the triangle. Predict which coordinate will be stored when you click **Export GLB**.
3. Move the downloaded file to `manual_tests/outputs/` if you want to keep it. Reopen it and find **Top vertex X** in the inspector. Reset to get the original triangle back.

## What the program does

`setVertex` edits the coordinates and updates the mesh bounds and normals. `frame` positions the preview camera around the object. Orbit controls update that camera while leaving the vertex array alone. The exporter receives a snapshot of the triangle, so your camera position and wireframe preference stay in the viewer.

A GLB starts with a 12-byte header identifying the format, version, and total length. The JSON chunk describes the objects and points to the data used to draw them. The binary chunk in this example contains the coordinates, triangle indices, colors, and normals. `inspectGlb` checks the container and reads the JSON. Three.js then decodes the mesh, and `summarize` counts what was loaded.

The scratchpad edits the generated sample. Imported GLBs and PNGs open for inspection. Later, the RealDenseFace bundle will include separate FLAME coefficients so you can regenerate the face after a parameter edit.

[GLB format specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#glb-file-format-specification)
