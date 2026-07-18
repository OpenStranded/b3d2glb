// b3d2glb — convert Blitz3D B3D models to glTF/GLB
// Copyright (C) 2024  Avenger Anubis (Ilya) <avenger.anubis@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::b3d::{AnimClip, JointInfo, MeshData, compute_world_matrices};
use crate::b3d_parser::{Brush, Texture};
use crate::cli::MaterialParams;
use crate::math::{mat4_inverse, swap_yz_pos, swap_yz_quat, quat_to_gltf, root_pos, root_quat};
use crate::texture::{load_texture, png_has_alpha, png_has_semi_transparent};

use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Write a binary .glb file with all data embedded (JSON + binary buffer).
pub fn write_glb(
    mesh: &MeshData,
    joints: &[JointInfo],
    clips: &[AnimClip],
    textures: &[Texture],
    brushes: &[Brush],
    model_name: &str,
    game_dir: &Path,
    tex_cache: &Path,
    out_path: &Path,
    material_params: Option<MaterialParams>,
    color_override: Option<[f32; 4]>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (root, bin, _) = build_gltf_inner(mesh, joints, clips, textures, brushes, model_name, game_dir, tex_cache, true, material_params, color_override)?;

    let json_str = serde_json::to_string(&root)?;
    let json_padded = pad_to_4(json_str.as_bytes());

    const HEADER_SIZE: u32 = 12;
    let total_len = HEADER_SIZE + 8 + json_padded.len() as u32 + 8 + bin.len() as u32;

    let mut glb = Vec::with_capacity(total_len as usize);
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&total_len.to_le_bytes());
    glb.extend_from_slice(&(json_padded.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"JSON");
    glb.extend_from_slice(&json_padded);
    glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    glb.extend_from_slice(b"BIN\0");
    glb.extend_from_slice(&bin);

    fs::write(out_path, glb)?;
    Ok(())
}

/// Write a .gltf file plus a separate .bin and texture files.
pub fn write_gltf_separate(
    mesh: &MeshData,
    joints: &[JointInfo],
    clips: &[AnimClip],
    textures: &[Texture],
    brushes: &[Brush],
    model_name: &str,
    game_dir: &Path,
    tex_cache: &Path,
    out_path: &Path,
    material_params: Option<MaterialParams>,
    color_override: Option<[f32; 4]>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (mut root, bin, image_infos) = build_gltf_inner(mesh, joints, clips, textures, brushes, model_name, game_dir, tex_cache, false, material_params, color_override)?;

    // Write binary buffer.
    let bin_path = out_path.with_extension("bin");
    fs::write(&bin_path, &bin)?;

    // Point the buffer to the external .bin file.
    let bin_name = bin_path.file_name().unwrap().to_string_lossy().to_string();
    if let Some(bufs) = root.get_mut("buffers").and_then(|v| v.as_array_mut()) {
        if let Some(buf) = bufs.get_mut(0).and_then(|v| v.as_object_mut()) {
            buf.insert("uri".into(), json!(bin_name));
        }
    }

    // Build separate image files + URI-based JSON.
    if !image_infos.is_empty() {
        let tex_dir = out_path.parent().unwrap_or(Path::new(".")).join("textures");
        let (images, gltf_textures) = build_image_uris(&image_infos, &tex_dir, model_name);
        root["images"] = json!(images);
        root["textures"] = json!(gltf_textures);
    }

    let json_str = serde_json::to_string_pretty(&root)?;
    fs::write(out_path, json_str)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// High-level Converter API
// ---------------------------------------------------------------------------

/// Builder-style converter for B3D → glTF/GLB conversion.
///
/// Provides the same functionality as the CLI but as a library API.
///
/// # Example
///
/// ```rust,no_run
/// use b3d2glb::writer::Converter;
/// use std::path::Path;
///
/// let b3d_data = std::fs::read("model.b3d").unwrap();
/// let glb_bytes = Converter::new("model", Path::new("/path/to/game"))
///     .glb(true)
///     .material(0.0, 0.9)
///     .convert_bytes(&b3d_data)
///     .unwrap();
/// ```
pub struct Converter {
    model_name: String,
    game_dir: std::path::PathBuf,
    tex_cache: std::path::PathBuf,
    glb_mode: bool,
    material: Option<crate::cli::MaterialParams>,
    color_override: Option<[f32; 4]>,
}

impl Converter {
    /// Create a new converter with default settings.
    ///
    /// * `model_name` — used for texture fallback and output naming.
    /// * `game_dir` — root directory for texture search (usually the game install dir).
    pub fn new(model_name: &str, game_dir: &std::path::Path) -> Self {
        let tex_cache = std::env::temp_dir().join("b3d2glb");
        Self {
            model_name: model_name.to_owned(),
            game_dir: game_dir.to_owned(),
            tex_cache,
            glb_mode: true,
            material: None,
            color_override: None,
        }
    }

    /// Set a custom texture cache directory.
    /// Defaults to a temporary directory.
    pub fn tex_cache(mut self, path: &std::path::Path) -> Self {
        self.tex_cache = path.to_owned();
        self
    }

    /// Enable/disable GLB mode.  Default: `true`.
    ///
    /// When `true`, the output is a single `.glb` file with all data embedded.
    /// When `false`, the output is a `.gltf` file with external `.bin` and textures.
    pub fn glb(mut self, glb: bool) -> Self {
        self.glb_mode = glb;
        self
    }

    /// Set metallic and roughness factors for all materials.
    /// Default: metallic=0.0, roughness=1.0.
    pub fn material(mut self, metallic: f32, roughness: f32) -> Self {
        self.material = Some(crate::cli::MaterialParams { metallic, roughness });
        self
    }

    /// Override the base color for materials without a texture.
    /// Default: `[0.8, 0.8, 0.8, 1.0]`.
    pub fn color_override(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.color_override = Some([r, g, b, a]);
        self
    }

    /// Convert raw B3D bytes to a complete GLB buffer (in memory).
    ///
    /// Returns the complete `.glb` file contents.
    pub fn convert_bytes(&self, b3d_data: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let (root, bin, _images) = self.build(b3d_data)?;

        // Assemble GLB: header + JSON chunk + BIN chunk
        let json_str = serde_json::to_string(&root)?;
        let json_bytes = pad_to_4(json_str.as_bytes());

        let mut glb = Vec::new();
        // GLB header (12 bytes)
        glb.extend_from_slice(b"glTF");
        glb.extend_from_slice(&2u32.to_le_bytes()); // version 2
        // Total length: header(12) + JSON chunk(8+json.len()) + BIN chunk(8+bin.len())
        let total_len = 12 + 8 + json_bytes.len() as u32 + 8 + bin.len() as u32;
        glb.extend_from_slice(&total_len.to_le_bytes());
        // JSON chunk
        glb.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"JSON");
        glb.extend_from_slice(&json_bytes);
        // BIN chunk
        glb.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        glb.extend_from_slice(b"BIN\x00");
        glb.extend_from_slice(&bin);

        Ok(glb)
    }

    /// Convert a B3D file to a GLB file on disk.
    pub fn convert_to_file(&self, input: &std::path::Path, output: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
        if self.glb_mode {
            let b3d_data = std::fs::read(input)?;
            let glb = self.convert_bytes(&b3d_data)?;
            std::fs::write(output, &glb)?;
            Ok(())
        } else {
            // For separate mode, delegate to the existing function.
            let b3d_data = std::fs::read(input)?;
            let b3d_parsed = crate::b3d_parser::B3D::read(&b3d_data)
                .map_err(|e| format!("parse error: {e}"))?;

            let vcount = b3d_parsed.node.mesh.vertices.vertices.len();
            if vcount == 0 {
                return Ok(());
            }

            let mut joints = Vec::new();
            let mut vertex_joint: Vec<Option<(usize, f32)>> = vec![None; vcount];
            crate::b3d::collect_joints(&b3d_parsed.node, None, &mut joints, &mut vertex_joint, vcount, true);
            let mesh = crate::b3d::collect_mesh(&b3d_parsed);
            let clips = crate::b3d::collect_anims(&b3d_parsed.node);

            write_gltf_separate(
                &mesh, &joints, &clips,
                &b3d_parsed.textures, &b3d_parsed.brushes,
                &self.model_name, &self.game_dir, &self.tex_cache,
                output,
                self.material,
                self.color_override,
            )
        }
    }

    /// Low-level: parse and build glTF data structures from B3D bytes.
    ///
    /// Returns the glTF JSON root, the binary buffer, and any embedded images.
    pub fn build(&self, b3d_data: &[u8]) -> Result<(serde_json::Value, Vec<u8>, Vec<ImageInfo>), Box<dyn std::error::Error>> {
        let b3d_parsed = crate::b3d_parser::B3D::read(b3d_data)
            .map_err(|e| format!("parse error: {e}"))?;

        let vcount = b3d_parsed.node.mesh.vertices.vertices.len();
        if vcount == 0 {
            return Err("model has no vertices".into());
        }

        let mut joints = Vec::new();
        let mut vertex_joint: Vec<Option<(usize, f32)>> = vec![None; vcount];
        crate::b3d::collect_joints(&b3d_parsed.node, None, &mut joints, &mut vertex_joint, vcount, true);
        let mut mesh = crate::b3d::collect_mesh(&b3d_parsed);
        // Apply joint data to mesh skin (collect_mesh doesn't do this).
        for (vi, j) in vertex_joint.iter().enumerate() {
            mesh.skin[vi] = j.as_ref().map(|(ji, w)| crate::b3d::BoneWeight {
                joint_idx: *ji as u32,
                weight: *w,
            });
        }
        let clips = crate::b3d::collect_anims(&b3d_parsed.node);

        // Ensure texture cache dir exists
        let _ = std::fs::create_dir_all(&self.tex_cache);

        build_gltf_inner(
            &mesh, &joints, &clips,
            &b3d_parsed.textures, &b3d_parsed.brushes,
            &self.model_name, &self.game_dir, &self.tex_cache,
            self.glb_mode,
            self.material,
            self.color_override,
        )
    }
}

// ---------------------------------------------------------------------------
// Internal: build the glTF JSON root + binary buffer
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn build_gltf_inner(
    mesh: &MeshData,
    joints: &[JointInfo],
    clips: &[AnimClip],
    b3d_textures: &[Texture],
    brushes: &[Brush],
    model_name: &str,
    game_dir: &Path,
    tex_cache: &Path,
    _embed_images: bool,
    material_params: Option<MaterialParams>,
    color_override: Option<[f32; 4]>,
) -> Result<(Value, Vec<u8>, Vec<ImageInfo>), Box<dyn std::error::Error>> {
    let vc = mesh.positions.len();
    let has_skin = !joints.is_empty() && mesh.skin.iter().any(|s| s.is_some());

    // --- materials ---------------------------------------------------------
    let brush_to_mat = build_brush_map(mesh);
    let (materials, image_infos, fallback_mat) = build_materials(
        &brush_to_mat, b3d_textures, brushes, model_name, game_dir, tex_cache, material_params, color_override,
    )?;

    // --- vertex buffer -----------------------------------------------------
    let mut bin = Vec::new();

    let pos_off = push_positions(&mut bin, &mesh.positions);
    let norm_off = push_normals(&mut bin, &mesh.normals);
    let uv_off = push_uvs(&mut bin, &mesh.uvs);

    let (joints_off, weights_off) = if has_skin {
        let jo = push_joints(&mut bin, &mesh.skin);
        let wo = push_weights(&mut bin, &mesh.skin);
        (Some(jo), Some(wo))
    } else {
        (None, None)
    };

    let idx_off = push_indices(&mut bin, &mesh.tri_groups);
    pad_to_4_in_place(&mut bin);

    let vc_u32 = vc as u32;

    // --- buffer views ------------------------------------------------------
    let mut bvs: Vec<Value> = vec![
        make_bv(0, pos_off, vc_u32 * 12, 12, 34962),
        make_bv(0, norm_off, vc_u32 * 12, 12, 34962),
        make_bv(0, uv_off, vc_u32 * 8, 8, 34962),
    ];

    let joints_bv = has_skin.then(|| {
        let i = bvs.len() as u32;
        bvs.push(make_bv(0, joints_off.unwrap(), vc_u32 * 8, 0, 34962));
        i
    });
    let weights_bv = has_skin.then(|| {
        let i = bvs.len() as u32;
        bvs.push(make_bv(0, weights_off.unwrap(), vc_u32 * 16, 0, 34962));
        i
    });

    let total_indices: u32 = mesh.tri_groups.iter().map(|t| t.indices.len() as u32).sum();
    let idx_bv = bvs.len() as u32;
    bvs.push(make_bv(0, idx_off, total_indices * 4, 0, 34963));

    // --- accessors ---------------------------------------------------------
    let mut accs: Vec<Value> = Vec::new();

    let (pos_min, pos_max) = calc_bounds(&mesh.positions);
    accs.push(json!({
        "bufferView": 0, "componentType": 5126, "count": vc_u32, "type": "VEC3",
        "min": [pos_min[0], pos_min[1], pos_min[2]],
        "max": [pos_max[0], pos_max[1], pos_max[2]],
    }));
    accs.push(json!({"bufferView": 1, "componentType": 5126, "count": vc_u32, "type": "VEC3"}));
    accs.push(json!({"bufferView": 2, "componentType": 5126, "count": vc_u32, "type": "VEC2"}));

    let joints_acc = has_skin.then(|| {
        let i = accs.len() as u32;
        accs.push(json!({"bufferView": joints_bv.unwrap(), "componentType": 5123, "count": vc_u32, "type": "VEC4"}));
        i
    });
    let weights_acc = has_skin.then(|| {
        let i = accs.len() as u32;
        accs.push(json!({"bufferView": weights_bv.unwrap(), "componentType": 5126, "count": vc_u32, "type": "VEC4"}));
        i
    });

    let base_idx_acc = accs.len() as u32;
    for (i, tg) in mesh.tri_groups.iter().enumerate() {
        let byte_start: u32 = mesh.tri_groups.iter().take(i).map(|t| t.indices.len() as u32 * 4).sum();
        accs.push(json!({
            "bufferView": idx_bv, "byteOffset": byte_start,
            "componentType": 5125, "count": tg.indices.len() as u32, "type": "SCALAR",
        }));
    }

    // --- nodes & scene -----------------------------------------------------
    let (gltf_nodes, scene_nodes) = build_node_hierarchy(joints, has_skin);

    // --- skin (IBM) --------------------------------------------------------
    let skins = build_skin_data(joints, has_skin, &mut bvs, &mut accs, &mut bin);

    // --- primitives --------------------------------------------------------
    let meshes = build_primitives(mesh, &brush_to_mat, fallback_mat,
        base_idx_acc, joints_acc, weights_acc);

    // --- animations --------------------------------------------------------
    let anim_acc_offset = accs.len() as u32;
    let animations = build_animations(clips, joints, anim_acc_offset, &mut bvs, &mut accs, &mut bin);

    // --- textures (embedded) -----------------------------------------------
    let (images, gltf_textures) = if !image_infos.is_empty() {
        build_image_json(&image_infos, &mut bvs, &mut bin)
    } else {
        (vec![], vec![])
    };

    // --- assemble root -----------------------------------------------------
    let mut root = json!({
        "asset": {"version": "2.0", "generator": "b3d2glb"},
        "scene": 0,
        "scenes": [{"nodes": scene_nodes}],
        "nodes": gltf_nodes,
        "meshes": meshes,
        "accessors": accs,
        "bufferViews": bvs,
        "buffers": [{"byteLength": bin.len() as u32}],
        "materials": materials,
    });

    if !skins.is_empty() { root["skins"] = json!(skins); }
    if !animations.is_empty() { root["animations"] = json!(animations); }
    if !images.is_empty() { root["images"] = json!(images); }
    if !gltf_textures.is_empty() { root["textures"] = json!(gltf_textures); }

    Ok((root, bin, image_infos))
}

// ---------------------------------------------------------------------------
// Buffer helpers
// ---------------------------------------------------------------------------

fn make_bv(buffer: u32, offset: usize, length: u32, stride: u32, target: u32) -> Value {
    let mut o = json!({
        "buffer": buffer,
        "byteOffset": offset,
        "byteLength": length,
        "target": target,
    });
    if stride > 0 {
        o["byteStride"] = json!(stride);
    }
    o
}

pub fn pad_to_4(data: &[u8]) -> Vec<u8> {
    let mut v = data.to_vec();
    while v.len() % 4 != 0 { v.push(0x20); }
    v
}

pub fn pad_to_4_in_place(data: &mut Vec<u8>) {
    while data.len() % 4 != 0 { data.push(0); }
}

// ---------------------------------------------------------------------------
// Vertex data writers
// ---------------------------------------------------------------------------

fn push_positions(bin: &mut Vec<u8>, positions: &[[f32; 3]]) -> usize {
    let off = bin.len();
    for p in positions {
        bin.extend_from_slice(&p[0].to_le_bytes());
        bin.extend_from_slice(&p[1].to_le_bytes());
        bin.extend_from_slice(&p[2].to_le_bytes());
    }
    off
}

fn push_normals(bin: &mut Vec<u8>, normals: &[[f32; 3]]) -> usize {
    let off = bin.len();
    for n in normals {
        bin.extend_from_slice(&n[0].to_le_bytes());
        bin.extend_from_slice(&n[1].to_le_bytes());
        bin.extend_from_slice(&n[2].to_le_bytes());
    }
    off
}

fn push_uvs(bin: &mut Vec<u8>, uvs: &[[f32; 2]]) -> usize {
    let off = bin.len();
    for uv in uvs {
        bin.extend_from_slice(&uv[0].to_le_bytes());
        bin.extend_from_slice(&uv[1].to_le_bytes());
    }
    off
}

fn push_joints(bin: &mut Vec<u8>, skin: &[Option<crate::b3d::BoneWeight>]) -> usize {
    let off = bin.len();
    for s in skin {
        let j = s.as_ref().map(|b| {
            debug_assert!(b.joint_idx <= u16::MAX as u32,
                "joint index {} exceeds u16 range (max 65535)", b.joint_idx);
            b.joint_idx as u16
        }).unwrap_or(0);
        bin.extend_from_slice(&j.to_le_bytes());
        bin.extend_from_slice(&0u16.to_le_bytes());
        bin.extend_from_slice(&0u16.to_le_bytes());
        bin.extend_from_slice(&0u16.to_le_bytes());
    }
    off
}

fn push_weights(bin: &mut Vec<u8>, skin: &[Option<crate::b3d::BoneWeight>]) -> usize {
    let off = bin.len();
    for s in skin {
        let w = s.as_ref().map(|b| b.weight).unwrap_or(0.0);
        bin.extend_from_slice(&w.to_le_bytes());
        bin.extend_from_slice(&0.0f32.to_le_bytes());
        bin.extend_from_slice(&0.0f32.to_le_bytes());
        bin.extend_from_slice(&0.0f32.to_le_bytes());
    }
    off
}

fn push_indices(bin: &mut Vec<u8>, tri_groups: &[crate::b3d::TriGroup]) -> usize {
    let off = bin.len();
    for tg in tri_groups {
        for &i in &tg.indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }
    }
    off
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------

fn build_brush_map(mesh: &MeshData) -> HashMap<u32, usize> {
    let mut map = HashMap::new();
    let mut sorted: Vec<u32> = mesh.tri_groups.iter().map(|t| t.brush_id).collect();
    sorted.sort();
    sorted.dedup();
    for (idx, bid) in sorted.iter().enumerate() {
        map.insert(*bid, idx);
    }
    map
}

pub struct ImageInfo {
    pub mime: String,
    pub data: Vec<u8>,
}

#[allow(clippy::too_many_arguments)]
fn build_materials(
    brush_to_mat: &HashMap<u32, usize>,
    b3d_textures: &[Texture],
    brushes: &[Brush],
    model_name: &str,
    game_dir: &Path,
    tex_cache: &Path,
    material_params: Option<MaterialParams>,
    color_override: Option<[f32; 4]>,
) -> Result<(Vec<Value>, Vec<ImageInfo>, usize), Box<dyn std::error::Error>> {
    // Use user-supplied material params, or sensible defaults.
    let metallic = material_params.map(|p| p.metallic).unwrap_or(0.0);
    let roughness = material_params.map(|p| p.roughness).unwrap_or(0.9);

    // When --color is given, use it as the fallback grey; otherwise keep [0.8,0.8,0.8,1].
    let fallback_color = color_override.unwrap_or([0.8, 0.8, 0.8, 1.0]);

    let mut materials: Vec<Value> = Vec::new();
    let mut image_infos: Vec<ImageInfo> = Vec::new();
    let fallback_mat = 0usize;

    // Determine sorted brush IDs from the map.
    let mut sorted: Vec<(u32, usize)> = brush_to_mat.iter()
        .map(|(&k, &v)| (k, v))
        .collect();
    sorted.sort_by_key(|&(k, _)| k);

    for &(brush_id, mat_idx) in &sorted {
        if (brush_id as usize) >= brushes.len() {
            continue;
        }
        let brush = &brushes[brush_id as usize];

        // Ensure the materials vec has room.
        materials.resize(mat_idx + 1, Value::Null);

        let tex_ref = brush.texture_id.first().and_then(|&tid| {
            let tid = tid as usize;
            (tid < b3d_textures.len()).then(|| &b3d_textures[tid])
        });

        let color = brush.color;

        // Determine alpha mode from B3D hints:
        // - flags & 2  = alpha channel → BLEND (smooth semi-transparency)
        // - flags & 4  = color key    → MASK  (hard cutoff)
        // - blend == 1 = alpha blend  → BLEND
        let mut alpha_mode: Option<&str> = tex_ref.and_then(|t| {
            if (t.flags & 2 != 0) || t.blend == 1 {
                Some("BLEND")
            } else if t.flags & 4 != 0 {
                Some("MASK")
            } else {
                None
            }
        });

        let mut mat_val = if let Some(tex) = tex_ref {
            let raw = tex.file
                .trim_start_matches(".\\")
                .trim_start_matches("./")
                .replace('\\', "/");
            let png_bytes = load_texture(&raw, game_dir, tex_cache);

            if let Some(bytes) = png_bytes {
                // Fallback: check actual pixel data when B3D flags don't indicate alpha.
                if alpha_mode.is_none() {
                    let has_alpha = png_has_alpha(&bytes);
                    if has_alpha {
                        // Determine MASK vs BLEND from pixel data:
                        // If any pixel has semi-transparent alpha (1-254), use BLEND.
                        // If all transparent pixels are fully transparent (0), use MASK.
                        alpha_mode = Some(if png_has_semi_transparent(&bytes) { "BLEND" } else { "MASK" });
                    }
                }
                let tex_idx = image_infos.len();
                image_infos.push(ImageInfo { mime: "image/png".into(), data: bytes });

                json!({
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                        "baseColorTexture": { "index": tex_idx },
                        "metallicFactor": metallic,
                        "roughnessFactor": roughness,
                    },
                    "doubleSided": true,
                })
            } else {
                json!({
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                        "metallicFactor": metallic,
                        "roughnessFactor": roughness,
                    },
                    "doubleSided": true,
                })
            }
        } else {
            json!({
                "pbrMetallicRoughness": {
                    "baseColorFactor": [color[0], color[1], color[2], color[3]],
                    "metallicFactor": metallic,
                    "roughnessFactor": roughness,
                },
                "doubleSided": true,
            })
        };

        if let Some(mode) = alpha_mode {
            if let Some(obj) = mat_val.as_object_mut() {
                obj.insert("alphaMode".into(), json!(mode));
                if mode == "MASK" {
                    obj.insert("alphaCutoff".into(), json!(0.5));
                }
            }
        }

        materials[mat_idx] = mat_val;
    }

    // Fallback: try a texture named after the model.
    if image_infos.is_empty() && !materials.is_empty() {
        if let Some(bytes) = load_texture(model_name, game_dir, tex_cache) {
            let tex_alpha = png_has_alpha(&bytes);
            let tex_idx = image_infos.len() as u32;
            image_infos.push(ImageInfo { mime: "image/png".into(), data: bytes });

            for mat in &mut materials {
                if let Some(obj) = mat.as_object_mut() {
                    if let Some(pbr) = obj.get_mut("pbrMetallicRoughness").and_then(|v| v.as_object_mut()) {
                        pbr.insert("baseColorFactor".into(), json!([1.0, 1.0, 1.0, 1.0]));
                        pbr.insert("baseColorTexture".into(), json!({"index": tex_idx}));
                    }
                    if tex_alpha {
                        obj.insert("alphaMode".into(), json!("MASK"));
                        obj.insert("alphaCutoff".into(), json!(0.5));
                    }
                }
            }
        }
    }

    // Ensure at least one material exists.
    if materials.is_empty() || materials.iter().all(|v| v.is_null()) {
        materials.clear();
        materials.push(json!({
            "pbrMetallicRoughness": {
                "baseColorFactor": fallback_color,
                "metallicFactor": metallic,
                "roughnessFactor": roughness,
            },
            "doubleSided": true,
        }));
    }

    Ok((materials, image_infos, fallback_mat))
}

// ---------------------------------------------------------------------------
// Nodes & scene
// ---------------------------------------------------------------------------

/// Build the glTF node hierarchy from B3D joints.
///
/// Each node stores the **bind-pose (rest) transform** in right-handed Y-up.
/// Animation channels (from `build_animations`) override these values per frame
/// via glTF's dispatch mechanism — at frame 0 the animation value equals the
/// bind pose, so the character appears at rest.
fn build_node_hierarchy(joints: &[JointInfo], has_skin: bool) -> (Vec<Value>, Vec<u32>) {
    // Match IDEAL node ordering: bones at 0..N-1, armature at N, ROOT at N+1.
    let n = joints.len();           // bone count
    let armature_idx = n as u32;    // armature slot
    let root_idx = (n + 1) as u32;  // ROOT slot

    // Step 1: bone nodes (indices 0..n-1).
    let mut gltf_nodes: Vec<Value> = joints.iter().enumerate().map(|(i, j)| {
        let (pos, rot_wxyz) = if j.parent.is_none() {
            (root_pos(j.position), root_quat(j.rotation))
        } else {
            (swap_yz_pos(j.position), swap_yz_quat(j.rotation))
        };

        let children: Vec<u32> = (0..joints.len())
            .filter(|&c| joints[c].parent == Some(i))
            .map(|c| c as u32)
            .collect();

        let mut node = json!({
            "name": j.name,
            "translation": [pos[0], pos[1], pos[2]],
            "rotation": quat_to_gltf(rot_wxyz),
        });
        if !children.is_empty() {
            node["children"] = json!(children);
        }
        node
    }).collect();

    // Step 2: armature node (index n).
    let root_children: Vec<u32> = (0..joints.len())
        .filter(|&i| joints[i].parent.is_none())
        .map(|i| i as u32)
        .collect();

    let mut armature = json!({"name": "armature"});
    if !root_children.is_empty() {
        armature["children"] = json!(root_children);
    }
    gltf_nodes.push(armature);

    // Step 3: ROOT node (index n+1) with mesh + skin.
    let mut root = json!({"name": "ROOT", "mesh": 0, "children": [armature_idx]});
    if has_skin {
        root["skin"] = json!(0);
    }
    gltf_nodes.push(root);

    let scene_nodes = vec![root_idx];
    (gltf_nodes, scene_nodes)
}

// ---------------------------------------------------------------------------
// Skin / IBM
// ---------------------------------------------------------------------------

/// Build glTF skin data: inverse bind matrices, buffer view, accessor, JSON.
fn build_skin_data(
    joints: &[JointInfo],
    has_skin: bool,
    bvs: &mut Vec<Value>,
    accs: &mut Vec<Value>,
    bin: &mut Vec<u8>,
) -> Vec<Value> {
    if !has_skin {
        return vec![];
    }
    let world_matrices = compute_world_matrices(joints);
    let ibm_off = bin.len();
    for world in &world_matrices {
        let inv = mat4_inverse(world);
        for col in 0..4 {
            bin.extend_from_slice(&inv[0][col].to_le_bytes());
            bin.extend_from_slice(&inv[1][col].to_le_bytes());
            bin.extend_from_slice(&inv[2][col].to_le_bytes());
            bin.extend_from_slice(&inv[3][col].to_le_bytes());
        }
    }
    pad_to_4_in_place(bin);

    let ibm_bv = bvs.len() as u32;
    bvs.push(make_bv(0, ibm_off, (joints.len() * 64) as u32, 0, 34962));

    let ibm_acc = accs.len() as u32;
    accs.push(json!({
        "bufferView": ibm_bv, "componentType": 5126,
        "count": joints.len() as u32, "type": "MAT4",
    }));

    let joint_ids: Vec<u32> = (0..joints.len() as u32).collect();
    vec![json!({
        "inverseBindMatrices": ibm_acc,
        "joints": joint_ids,
    })]
}

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

/// Build glTF mesh primitives from triangle groups.
fn build_primitives(
    mesh: &MeshData,
    brush_to_mat: &HashMap<u32, usize>,
    fallback_mat: usize,
    base_idx_acc: u32,
    joints_acc: Option<u32>,
    weights_acc: Option<u32>,
) -> Vec<Value> {
    let mut primitives = Vec::new();
    for (i, tg) in mesh.tri_groups.iter().enumerate() {
        let mat = brush_to_mat.get(&tg.brush_id).copied().unwrap_or(fallback_mat);
        let mut prim = json!({
            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
            "indices": base_idx_acc + i as u32,
            "material": mat,
        });

        if let (Some(ja), Some(wa)) = (joints_acc, weights_acc) {
            if let Some(attrs) = prim.pointer_mut("/attributes").and_then(|v| v.as_object_mut()) {
                attrs.insert("JOINTS_0".into(), json!(ja));
                attrs.insert("WEIGHTS_0".into(), json!(wa));
            }
        }

        primitives.push(prim);
    }
    vec![json!({"primitives": primitives})]
}

// ---------------------------------------------------------------------------
// Animations
// ---------------------------------------------------------------------------

fn build_animations(
    clips: &[AnimClip],
    joints: &[JointInfo],
    acc_start: u32,
    bvs: &mut Vec<Value>,
    accs: &mut Vec<Value>,
    bin: &mut Vec<u8>,
) -> Vec<Value> {
    let mut gltf_anims: Vec<Value> = Vec::new();
    let mut acc_counter = acc_start;

    for clip in clips {
        let fps = if clip.fps > 0.0 { clip.fps } else { 30.0 };
        let mut channels: Vec<Value> = Vec::new();
        let mut samplers: Vec<Value> = Vec::new();

        for (ji, joint) in joints.iter().enumerate() {
            if joint.keys.is_empty() { continue; }

            let all_keys: Vec<&(u32, [f32; 3], [f32; 3], [f32; 4])> = joint.keys.iter()
                .filter(|(frame, _, _, _)| *frame >= clip.first_frame && *frame <= clip.last_frame)
                .collect();
            if all_keys.is_empty() { continue; }

            // Split keys by channel type.  After the b3d crate fix (merge of
            // multiple KEYS chunks), position-only keys have rotation=(0,0,0,0)
            // and rotation-only keys have position=(0,0,0) as defaults.
            let pos_keys: Vec<_> = all_keys.iter()
                .filter(|(_, p, _, _)| p[0] != 0.0 || p[1] != 0.0 || p[2] != 0.0)
                .collect();
            let scl_keys: Vec<_> = all_keys.iter()
                .filter(|(_, _, s, _)| s[0] != 0.0 || s[1] != 0.0 || s[2] != 0.0)
                .collect();
            let rot_keys: Vec<_> = all_keys.iter()
                .filter(|(_, _, _, r)| r[0] != 0.0 || r[1] != 0.0 || r[2] != 0.0 || r[3] != 0.0)
                .collect();

            // Normalize time so the first keyframe is at t=0,
            // matching the Blender plugin (k.frame - pos_keys[0].frame).
            let first_frame = all_keys.iter().map(|(f,_,_,_)| *f).min().unwrap_or(0);
            let last_frame = all_keys.iter().map(|(f,_,_,_)| *f).max().unwrap_or(0);
            let t0 = first_frame as f32 / fps;
            let t1 = last_frame as f32 / fps;

            // Helper to emit one channel (with or without actual keys).
            // If `keys` is empty, two dummy keys (t0/t1, default_val) are used
            // to match the Blender exporter's "always-emit-TRS" convention.
            macro_rules! emit_chan {
                ($keys:expr, $default:expr, $path:expr, $elem_size:expr, $ty:expr, $enco:expr) => {{
                    let kk = $keys;
                    let kc = if kk.is_empty() { 2usize } else { kk.len() };
                    let es = $elem_size;

                    // Time accessor.
                    let to = bin.len();
                    if !kk.is_empty() {
                        for (frame, _, _, _) in &kk {
                            let t = (*frame - first_frame) as f32 / fps;
                            bin.extend_from_slice(&t.to_le_bytes());
                        }
                    } else {
                        bin.extend_from_slice(&t0.to_le_bytes());
                        bin.extend_from_slice(&t1.to_le_bytes());
                    }
                    let tb = bvs.len() as u32;
                    bvs.push(make_bv(0, to, (kc as u32 * 4) as u32, 0, 34962));
                    let ta = acc_counter;
                    accs.push(json!({"bufferView": tb, "componentType": 5126, "count": kc as u32, "type": "SCALAR"}));
                    acc_counter += 1;

                    // Value accessor.
                    let vo = bin.len();
                    if !kk.is_empty() {
                        for k in &kk { $enco(bin, k); }
                    } else {
                        bin.extend_from_slice(&$default[0].to_le_bytes());
                        bin.extend_from_slice(&$default[1].to_le_bytes());
                        bin.extend_from_slice(&$default[2].to_le_bytes());
                        if es > 12 {
                            bin.extend_from_slice(&$default[3].to_le_bytes());
                        }
                        // duplicate for t1
                        bin.extend_from_slice(&$default[0].to_le_bytes());
                        bin.extend_from_slice(&$default[1].to_le_bytes());
                        bin.extend_from_slice(&$default[2].to_le_bytes());
                        if es > 12 {
                            bin.extend_from_slice(&$default[3].to_le_bytes());
                        }
                    }
                    let vb = bvs.len() as u32;
                    bvs.push(make_bv(0, vo, (kc as u32 * es) as u32, es, 34962));
                    let va = acc_counter;
                    accs.push(json!({"bufferView": vb, "componentType": 5126, "count": kc as u32, "type": $ty}));
                    acc_counter += 1;

                    let si = samplers.len() as u32;
                    samplers.push(json!({"input": ta, "output": va, "interpolation": "LINEAR"}));
                    // Node 0 = root, bones start at 1.
                    let node_i: u32 = ji as u32;
                    channels.push(json!({"sampler": si, "target": {"node": node_i, "path": $path}}));
                }};
            }

            // Position
            let def_pos = if joint.parent.is_none() { root_pos(joint.position) } else { swap_yz_pos(joint.position) };
            let encode_pos = if joint.parent.is_none() {
                |bin: &mut Vec<u8>, k: &&(u32, [f32;3], [f32;3], [f32;4])| {
                    let cp = root_pos(k.1);
                    bin.extend_from_slice(&cp[0].to_le_bytes());
                    bin.extend_from_slice(&cp[1].to_le_bytes());
                    bin.extend_from_slice(&cp[2].to_le_bytes());
                }
            } else {
                |bin: &mut Vec<u8>, k: &&(u32, [f32;3], [f32;3], [f32;4])| {
                    let cp = swap_yz_pos(k.1);
                    bin.extend_from_slice(&cp[0].to_le_bytes());
                    bin.extend_from_slice(&cp[1].to_le_bytes());
                    bin.extend_from_slice(&cp[2].to_le_bytes());
                }
            };
            emit_chan!(pos_keys, def_pos, "translation", 12u32, "VEC3", encode_pos);
            // Scale  (always identity)
            emit_chan!(scl_keys, [1f32,1f32,1f32,0f32], "scale", 12u32, "VEC3", |bin: &mut Vec<u8>, k: &&(u32, [f32;3], [f32;3], [f32;4])| {
                let s = k.2;
                bin.extend_from_slice(&s[0].to_le_bytes());
                bin.extend_from_slice(&s[1].to_le_bytes());
                bin.extend_from_slice(&s[2].to_le_bytes());
            });
            // Rotation
            let def_rot = quat_to_gltf(
                if joint.parent.is_none() { root_quat(joint.rotation) } else { swap_yz_quat(joint.rotation) }
            );
            let encode_rot = if joint.parent.is_none() {
                |bin: &mut Vec<u8>, k: &&(u32, [f32;3], [f32;3], [f32;4])| {
                    let q = quat_to_gltf(root_quat(k.3));
                    bin.extend_from_slice(&q[0].to_le_bytes());
                    bin.extend_from_slice(&q[1].to_le_bytes());
                    bin.extend_from_slice(&q[2].to_le_bytes());
                    bin.extend_from_slice(&q[3].to_le_bytes());
                }
            } else {
                |bin: &mut Vec<u8>, k: &&(u32, [f32;3], [f32;3], [f32;4])| {
                    let q = quat_to_gltf(swap_yz_quat(k.3));
                    bin.extend_from_slice(&q[0].to_le_bytes());
                    bin.extend_from_slice(&q[1].to_le_bytes());
                    bin.extend_from_slice(&q[2].to_le_bytes());
                    bin.extend_from_slice(&q[3].to_le_bytes());
                }
            };
            emit_chan!(rot_keys, def_rot, "rotation", 16u32, "VEC4", encode_rot);
        }

        pad_to_4_in_place(bin);

        if !channels.is_empty() {
            gltf_anims.push(json!({
                "name": clip.name,
                "channels": channels,
                "samplers": samplers,
            }));
        }
    }

    gltf_anims
}

// ---------------------------------------------------------------------------
// Image / texture JSON (embedded or external)
// ---------------------------------------------------------------------------

fn build_image_json(
    image_infos: &[ImageInfo],
    bvs: &mut Vec<Value>,
    bin: &mut Vec<u8>,
) -> (Vec<Value>, Vec<Value>) {
    pad_to_4_in_place(bin);

    let mut images = Vec::new();
    let mut textures = Vec::new();

    for info in image_infos {
        let img_off = bin.len();
        bin.extend_from_slice(&info.data);
        pad_to_4_in_place(bin);

        let bv_idx = bvs.len() as u32;
        bvs.push(json!({
            "buffer": 0,
            "byteOffset": img_off,
            "byteLength": info.data.len() as u32,
        }));

        images.push(json!({
            "mimeType": info.mime,
            "bufferView": bv_idx,
        }));
        textures.push(json!({"source": textures.len() as u32}));
    }

    (images, textures)
}

fn build_image_uris(
    image_infos: &[ImageInfo],
    tex_out_dir: &Path,
    model_name: &str,
) -> (Vec<Value>, Vec<Value>) {
    let mut images = Vec::new();
    let mut textures = Vec::new();

    for (i, info) in image_infos.iter().enumerate() {
        let fname = format!("{model_name}_tex{i}.png");
        let fpath = tex_out_dir.join(&fname);

        // Write the PNG file.
        let _ = std::fs::write(&fpath, &info.data);

        images.push(json!({
            "mimeType": info.mime,
            "uri": format!("textures/{fname}"),
        }));
        textures.push(json!({"source": textures.len() as u32}));
    }

    (images, textures)
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::b3d::{JointInfo, MeshData, TriGroup};

    #[test]
    fn test_calc_bounds_basic() {
        let positions = vec![[1.0, 2.0, 3.0], [4.0, -5.0, 6.0], [-7.0, 8.0, -9.0]];
        let (min, max) = calc_bounds(&positions);
        assert_eq!(min, [-7.0, -5.0, -9.0]);
        assert_eq!(max, [4.0, 8.0, 6.0]);
    }

    #[test]
    fn test_calc_bounds_single() {
        let positions = vec![[10.0, 20.0, 30.0]];
        let (min, max) = calc_bounds(&positions);
        assert_eq!(min, max);
        assert_eq!(min, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_build_brush_map_empty() {
        let md = MeshData {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            tri_groups: vec![],
            skin: vec![],
        };
        let map = build_brush_map(&md);
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_build_brush_map_sorted() {
        let md = MeshData {
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            tri_groups: vec![
                TriGroup { brush_id: 5, indices: vec![0,1,2] },
                TriGroup { brush_id: 2, indices: vec![3,4,5] },
                TriGroup { brush_id: 5, indices: vec![6,7,8] },
            ],
            skin: vec![],
        };
        let map = build_brush_map(&md);
        assert_eq!(map.len(), 2); // 2 unique brush IDs
        assert_eq!(map.get(&2), Some(&0));
        assert_eq!(map.get(&5), Some(&1));
    }

    #[test]
    fn test_build_node_hierarchy_no_joints() {
        let (nodes, scene) = build_node_hierarchy(&[], false);
        // n=0 → armature at 0, ROOT at 1 (2 nodes)
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].get("name").and_then(|v| v.as_str()), Some("armature"));
        assert_eq!(nodes[1].get("name").and_then(|v| v.as_str()), Some("ROOT"));
        assert_eq!(nodes[1].get("mesh").and_then(|v| v.as_u64()), Some(0));
        assert!(nodes[1].get("skin").is_none());
        assert_eq!(scene, vec![1u32]);
    }

    #[test]
    fn test_build_node_hierarchy_single_joint() {
        let joints = vec![
            JointInfo {
                name: "hip".into(),
                position: [0.0, 10.0, -5.0],
                scale: [1.0, 1.0, 1.0],
                rotation: [1.0, 0.0, 0.0, 0.0],
                parent: None,
                keys: vec![],
            },
        ];
        let (nodes, scene) = build_node_hierarchy(&joints, true);
        // 1 bone + 1 armature + 1 ROOT = 3 nodes
        assert_eq!(nodes.len(), 3);
        // node 0 = hip
        assert_eq!(nodes[0].get("name").and_then(|v| v.as_str()), Some("hip"));
        // node 1 = armature
        assert_eq!(nodes[1].get("name").and_then(|v| v.as_str()), Some("armature"));
        assert_eq!(nodes[1]["children"][0].as_u64(), Some(0));
        // node 2 = ROOT with mesh + skin + armature child
        assert_eq!(nodes[2].get("name").and_then(|v| v.as_str()), Some("ROOT"));
        assert_eq!(nodes[2]["children"][0].as_u64(), Some(1));
        assert_eq!(nodes[2]["mesh"].as_u64(), Some(0));
        assert_eq!(nodes[2]["skin"].as_u64(), Some(0));
        // scene has just ROOT
        assert_eq!(scene.len(), 1);
        assert_eq!(scene[0], 2);
    }
}

fn calc_bounds(positions: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX, f32::MAX, f32::MAX];
    let mut max = [f32::MIN, f32::MIN, f32::MIN];
    for p in positions {
        if p[0] < min[0] { min[0] = p[0]; }
        if p[1] < min[1] { min[1] = p[1]; }
        if p[2] < min[2] { min[2] = p[2]; }
        if p[0] > max[0] { max[0] = p[0]; }
        if p[1] > max[1] { max[1] = p[1]; }
        if p[2] > max[2] { max[2] = p[2]; }
    }
    (min, max)
}
