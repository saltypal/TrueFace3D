import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFExporter } from "three/addons/exporters/GLTFExporter.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { inspectGlb } from "./glb.js";

const byId = (id) => document.getElementById(id);
const viewport = byId("viewport");
const image = byId("image-preview");
const slider = byId("vertex-x");
const wireframe = byId("wireframe");
const status = (message) => { byId("status").textContent = message; };
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x192533);
const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 1000);
scene.add(new THREE.HemisphereLight(0xffffff, 0x556677, 2.5));

let renderer, controls;
try {
  // The preview only renders on interaction and resize, so retain the last frame.
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  viewport.append(renderer.domElement);
  controls = new OrbitControls(camera, renderer.domElement);
} catch {
  status("WebGL is unavailable. Open a PNG to preview a saved render.");
}

const positions = new THREE.BufferAttribute(new Float32Array([-1, -.7, 0, 1, -.7, 0, 0, .95, 0]), 3);
const geometry = new THREE.BufferGeometry();
geometry.setAttribute("position", positions);
geometry.setAttribute("color", new THREE.Float32BufferAttribute([.05, .6, .9, .1, .95, .65, 1, .4, .15], 3));
geometry.setIndex([0, 1, 2]);
geometry.computeVertexNormals();
const triangle = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
triangle.name = "Scratchpad triangle";
triangle.material.name = "Vertex colors";
const exporter = new GLTFExporter();

let active, imageUrl, request = 0, exporting = false;
function updateButtons() {
  slider.disabled = wireframe.disabled = !renderer || active !== triangle;
  byId("export").disabled = slider.disabled || exporting;
}

function render() {
  if (!renderer || !image.hidden) return;
  renderer.render(scene, camera);
}

function frame(object) {
  const bounds = new THREE.Box3().setFromObject(object);
  const sphere = bounds.getBoundingSphere(new THREE.Sphere());
  if (!Number.isFinite(sphere.radius) || sphere.radius <= 0) throw new Error("The asset has no visible mesh bounds.");
  if (!controls) return;
  const distance = sphere.radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * Math.max(1, 1 / camera.aspect) * 1.15;
  camera.near = Math.max(sphere.radius / 1000, 0.000001);
  camera.far = distance + sphere.radius * 100;
  camera.updateProjectionMatrix();
  controls.target.copy(sphere.center);
  camera.position.copy(sphere.center).add(new THREE.Vector3(0, 0, distance));
  controls.update();
}

function showMesh(object) {
  frame(object);
  if (active) scene.remove(active);
  active = object;
  scene.add(object);
  image.hidden = true;
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = null;
  image.removeAttribute("src");
  if (renderer) renderer.domElement.hidden = false;
  updateButtons();
  render();
}

function setVertex(value) {
  positions.setX(2, value);
  positions.needsUpdate = true;
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  slider.value = value;
  byId("vertex-value").textContent = value.toFixed(2);
  render();
}

function summarize(bytes, mesh) {
  const { json, chunks } = inspectGlb(bytes);
  let meshes = 0, triangles = 0;
  mesh.traverse((node) => {
    if (!node.isMesh) return;
    meshes++;
    triangles += (node.geometry.index?.count ?? node.geometry.attributes.position.count) / 3;
  });
  const materialNames = (json.materials ?? []).map((item) => item.name || "(unnamed)").join(", ");
  const top = mesh.getObjectByName("Scratchpad_triangle") ?? mesh.getObjectByName("Scratchpad triangle");
  const vertex = top?.isMesh && top.geometry.attributes.position.count === 3
    ? "Top vertex X: " + top.geometry.attributes.position.getX(2).toFixed(2) : "";
  return [
    "GLB 2.0 · " + bytes.byteLength + " bytes",
    ...chunks, "Meshes: " + meshes, "Triangles: " + triangles,
    "Materials: " + (materialNames || "none"), vertex,
  ].filter(Boolean).join("\n");
}

slider.addEventListener("input", () => setVertex(Number(slider.value)));
wireframe.addEventListener("change", () => { triangle.material.wireframe = wireframe.checked; render(); });
byId("reset").addEventListener("click", () => {
  request++;
  setVertex(0);
  wireframe.checked = triangle.material.wireframe = false;
  showMesh(triangle);
  byId("inspection").textContent = "Export or open a GLB to inspect it.";
  if (renderer) status("Triangle reset. Drag to orbit and scroll to zoom.");
});

byId("export").addEventListener("click", async () => {
  const version = request;
  exporting = true;
  updateButtons();
  const snapshot = new THREE.Mesh(geometry.clone(), triangle.material.clone());
  snapshot.name = triangle.name;
  snapshot.material.wireframe = false;
  try {
    const bytes = await exporter.parseAsync(snapshot, { binary: true });
    const url = URL.createObjectURL(new Blob([bytes], { type: "model/gltf-binary" }));
    const link = Object.assign(document.createElement("a"), { href: url, download: "scratchpad-triangle.glb" });
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    if (version === request) {
      byId("inspection").textContent = summarize(bytes, snapshot);
      status("GLB exported. Open the downloaded file to check the edit.");
    }
  } catch (error) {
    if (version === request) status("Export failed: " + error.message);
  } finally {
    snapshot.geometry.dispose();
    snapshot.material.dispose();
    exporting = false;
    updateButtons();
  }
});

byId("open-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  event.target.value = "";
  if (!file) return;
  const version = ++request;
  try {
    if (file.name.toLowerCase().endsWith(".png")) {
      const url = URL.createObjectURL(file);
      const decoded = new Image();
      decoded.src = url;
      try { await decoded.decode(); }
      catch (error) { URL.revokeObjectURL(url); throw error; }
      if (version !== request) { URL.revokeObjectURL(url); return; }
      if (imageUrl) URL.revokeObjectURL(imageUrl);
      imageUrl = url;
      image.src = url;
      image.hidden = false;
      if (renderer) renderer.domElement.hidden = true;
      if (active) scene.remove(active);
      active = null;
      updateButtons();
      byId("inspection").textContent = "PNG preview. Open a GLB to inspect its contents.";
      status("Opened " + file.name);
      return;
    }
    const bytes = await file.arrayBuffer();
    const { json } = inspectGlb(bytes);
    if ([...(json.buffers ?? []), ...(json.images ?? [])].some((item) => item.uri && !item.uri.startsWith("data:"))) {
      throw new Error("Choose a self-contained GLB with its resources embedded.");
    }
    const { scene: imported } = await new GLTFLoader().parseAsync(bytes, "");
    if (!json.materials?.length) {
      const matte = new THREE.MeshStandardMaterial({ color: 0xc5d0d9, metalness: 0, roughness: .9 });
      imported.traverse((node) => { if (node.isMesh) node.material = matte; });
    }
    const report = summarize(bytes, imported);
    if (version !== request) return;
    showMesh(imported);
    byId("inspection").textContent = report;
    status("Opened " + file.name + (renderer ? ". Reset triangle to return to editing." : ". WebGL is unavailable."));
  } catch (error) {
    if (version === request) status("Could not open file: " + error.message);
  }
});

if (renderer) {
  controls.addEventListener("change", render);
  new ResizeObserver(() => {
    const { clientWidth, clientHeight } = viewport;
    if (!clientWidth || !clientHeight) return;
    renderer.setSize(clientWidth, clientHeight, false);
    camera.aspect = clientWidth / clientHeight;
    camera.updateProjectionMatrix();
    render();
  }).observe(viewport);
}
byId("reset").click();
