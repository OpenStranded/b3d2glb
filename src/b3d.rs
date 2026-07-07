pub use b3d::B3D;
use b3d::Node;
use crate::math::Mat4;
use crate::math;

/// Per-vertex skinning data (B3D stores at most one bone per vertex).
#[derive(Debug, Clone)]
pub struct BoneWeight {
    pub joint_idx: u32,
    pub weight: f32,
}

/// A group of triangle indices sharing a material (brush).
#[derive(Debug, Clone)]
pub struct TriGroup {
    pub brush_id: u32,
    pub indices: Vec<u32>,
}

/// Extracted mesh data from a B3D node.
#[derive(Debug, Clone)]
pub struct MeshData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub tri_groups: Vec<TriGroup>,
    pub skin: Vec<Option<BoneWeight>>,
}

/// One joint (bone) in the skeleton hierarchy.
#[derive(Debug, Clone)]
pub struct JointInfo {
    pub name: String,
    /// Local bind-pose translation (B3D coordinates).
    pub position: [f32; 3],
    /// Local bind-pose scale.
    pub scale: [f32; 3],
    /// Local bind-pose rotation quaternion (B3D: [w, x, y, z]).
    pub rotation: [f32; 4],
    /// Index of parent joint in the flattened array, or `None` for root.
    pub parent: Option<usize>,
    /// Keyframes: `(frame, position, scale, rotation)`.
    pub keys: Vec<(u32, [f32; 3], [f32; 3], [f32; 4])>,
}

/// A named animation clip derived from B3D sequences.
#[derive(Debug, Clone)]
pub struct AnimClip {
    pub name: String,
    pub fps: f32,
    pub first_frame: u32,
    pub last_frame: u32,
}

/// Traverse the B3D node tree, collecting joints and vertex-to-joint mapping.
///
/// Set `is_root = true` for the initial call (the mesh node). The root node is
/// NOT added to the joint list – it is the skinned mesh, not a deforming bone.
pub fn collect_joints(
    node: &Node,
    parent: Option<usize>,
    joints: &mut Vec<JointInfo>,
    vertex_joint: &mut Vec<Option<(usize, f32)>>,
    vcount: usize,
    is_root: bool,
) {
    let idx = if is_root { 0 } else { joints.len() };
    if !is_root {
        let keys: Vec<_> = node.keys.iter().map(|k| {
            (k.frame, k.position, k.scale, k.rotation)
        }).collect();

        joints.push(JointInfo {
            name: node.name.clone(),
            position: node.position,
            scale: node.scale,
            rotation: node.rotation,
            parent,
            keys,
        });
    }

    for b in &node.bones {
        let vi = b.vertex_id as usize;
        if vi < vcount {
            // Accumulate weights (a vertex may be assigned to multiple bones).
            // For now we keep only the first (or last) assignment; a proper fix
            // would store all weights and let the writer pick up to 4.
            vertex_joint[vi] = Some((idx, b.weight));
        }
    }

    for child in &node.children {
        // Root's children get parent=None since the mesh isn't a joint.
        let child_parent = if is_root { None } else { Some(idx) };
        collect_joints(child, child_parent, joints, vertex_joint, vcount, false);
    }
}

/// Collect named animation clips from the B3D node tree.
pub fn collect_anims(node: &Node) -> Vec<AnimClip> {
    let mut anims = Vec::new();
    let fps = if node.animation.fps > 0.0 { node.animation.fps } else { 30.0 };

    if !node.sequences.is_empty() {
        for seq in &node.sequences {
            anims.push(AnimClip {
                name: seq.name.clone(),
                fps,
                first_frame: seq.first_frame,
                last_frame: seq.last_frame,
            });
        }
    } else if node.animation.frames > 1 {
        anims.push(AnimClip {
            name: "default".into(),
            fps,
            first_frame: 0,
            last_frame: node.animation.frames.saturating_sub(1),
        });
    }

    anims
}

/// Extract mesh geometry from a parsed B3D file.
pub fn collect_mesh(b3d: &B3D) -> MeshData {
    let verts = &b3d.node.mesh.vertices;
    let vc = verts.vertices.len();

    let mut positions = Vec::with_capacity(vc);
    let mut normals = Vec::with_capacity(vc);
    let mut uvs = Vec::with_capacity(vc);

    for v in &verts.vertices {
        // Mesh vertices are in root-node local space, same as root joints.
        // Convert from B3D (left-handed Y-up) to glTF (right-handed Y-up)
        // by negating Z: [x, y, -z]. This matches root_pos for root bones.
        positions.push([v.position[0], v.position[1], -v.position[2]]);
        normals.push([v.normal[0], v.normal[1], -v.normal[2]]);
        uvs.push([v.tex_coords[0], v.tex_coords[1]]);
    }

    let mut tri_groups = Vec::new();
    for tris in &b3d.node.mesh.triangles {
        let mut indices = Vec::with_capacity(tris.indices.len() * 3);
        for tri in &tris.indices {
            indices.push(tri[0]);
            indices.push(tri[2]);
            indices.push(tri[1]);
        }
        tri_groups.push(TriGroup { brush_id: tris.brush_id, indices });
    }

    let skin = (0..vc).map(|_| None).collect();
    MeshData { positions, normals, uvs, tri_groups, skin }
}

/// Compute the world-space matrix for a joint in right-handed Y-up (glTF space).
///
/// B3D stores bind-pose TRS in left-handed Y-up.
/// Conversion matches `build_node_hierarchy`: root bones use `[x, y, -z]` + -90° X,
/// children use `[x, z, y]` / `[w, x, z, y]`.
pub fn compute_world_matrix(joints: &[JointInfo], idx: usize) -> Mat4 {
    let scale = joints[idx].scale;
    let (pos, rot) = if joints[idx].parent.is_none() {
        (math::root_pos(joints[idx].position), math::root_quat(joints[idx].rotation))
    } else {
        (math::neg_z_pos(joints[idx].position), math::neg_z_quat(joints[idx].rotation))
    };
    let local = math::b3d_to_mat4(pos, scale, rot);
    match joints[idx].parent {
        Some(p) => math::mat4_mul(&compute_world_matrix(joints, p), &local),
        None => local,
    }
}
