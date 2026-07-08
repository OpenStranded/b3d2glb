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

use std::io::Cursor;
use std::io::{Read, Seek};
use byteorder::{ReadBytesExt, LittleEndian};
#[cfg(test)] use byteorder::WriteBytesExt;

mod utils;

use utils::*;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]
    IO(#[from] std::io::Error),
    #[error(transparent)]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("Invalid Chunk: {0}")]
    InvalidChunk(Chunk),
}

#[derive(Debug, Clone)]
pub struct Texture {
    pub file: String,
    pub flags: u32,
    pub blend: u32,
    pub position: Vec2,
    pub scale: Vec2,
    pub rotation: f32,
}

impl Texture {
    pub fn read<T>(data: &mut T) -> Result<Self, Error>
    where
        T: Read + Seek
    {
        let file = read_null_term_string(data);
        let flags = data.read_u32::<LittleEndian>()?;
        let blend = data.read_u32::<LittleEndian>()?;
        let mut position = [0.0; 2];
        data.read_f32_into::<LittleEndian>(&mut position)?;
        let mut scale = [0.0; 2];
        data.read_f32_into::<LittleEndian>(&mut scale)?;
        let rotation = data.read_f32::<LittleEndian>()?;

        Ok(Self {
            file,
            flags,
            blend,
            position,
            scale,
            rotation,
        })
    }
}

#[derive(Debug)]
pub struct Brush {
    pub name: String,
    pub color: Vec4,
	pub shininess: f32,
	pub blend: u32,
	pub fx: u32,
	pub texture_id: Vec<u32>,
}

impl Brush {
    pub fn read<T>(data: &mut T, n_texs: usize) -> Result<Self, Error>
    where
        T: Read + Seek
    {
        let name = read_null_term_string(data);
        let mut color = [0.0; 4];
        data.read_f32_into::<LittleEndian>(&mut color)?;
        let shininess = data.read_f32::<LittleEndian>()?;
        let blend = data.read_u32::<LittleEndian>()?;
        let fx = data.read_u32::<LittleEndian>()?;

        let mut texture_id = vec![];

        for _ in 0..n_texs {
            texture_id.push(data.read_u32::<LittleEndian>()?);
        }

        Ok(Self {
            name,
            color,
            shininess,
            blend,
            fx,
            texture_id,
        })
    }
}

#[derive(Debug, Default)]
pub struct Vertice {
    pub position: Vec3,
    pub normal: Vec3,
    pub color: Vec4,
    pub tex_coords: Vec2,
}

#[derive(Debug, Default)]
pub struct Verts {
    pub flags: u32,
    pub tex_coord_sets: u32,
    pub tex_coord_set_size: u32,
    pub vertices: Vec<Vertice>,
}

impl Verts {
    pub fn read<T>(data: &mut T, next: u64) -> Result<Self, Error>
    where
        T: Read + Seek
    {
        let flags = data.read_u32::<LittleEndian>()?;
        let tex_coord_sets = data.read_u32::<LittleEndian>()?;
        let tex_coord_set_size = data.read_u32::<LittleEndian>()?;

        let mut vertices: Vec<Vertice> = Vec::new();

        while eof(data, next)? {
            let mut position = [0.0; 3];
            data.read_f32_into::<LittleEndian>(&mut position)?;
            let mut normal = [0.0; 3];
            if flags & 1 != 0 {
                data.read_f32_into::<LittleEndian>(&mut normal)?;
            }
            let mut color = [0.0; 4];
            if flags & 2 != 0 {
                data.read_f32_into::<LittleEndian>(&mut color)?;
            }
            let mut tex_coords = [0.0; 2];
            data.read_f32_into::<LittleEndian>(&mut tex_coords)?;

            vertices.push(Vertice {
                position,
                normal,
                color,
                tex_coords,
            });
        }

        Ok(Self {
            flags,
            tex_coord_sets,
            tex_coord_set_size,
            vertices,
        })
    }
}

#[derive(Debug)]
pub struct Tris {
    pub brush_id: u32,
    pub indices: Vec<[u32; 3]>,
}

impl Tris {
    pub fn read<T>(data: &mut T, next: u64) -> Result<Self, Error>
    where
        T: Read + Seek
    {
        let brush_id = data.read_u32::<LittleEndian>()?;
        let mut indices = Vec::new();

        while eof(data, next)? {
            let mut face = [0; 3];
            data.read_u32_into::<LittleEndian>(&mut face)?;
            indices.push(face);
        }

        Ok(Self {
            brush_id,
            indices,
        })
    }
}

#[derive(Debug, Default)]
pub struct Mesh {
    pub brush_id: u32,
    pub vertices: Verts,
    pub triangles: Vec<Tris>,
}

impl Mesh {
    pub fn read<T>(data: &mut T, next: u64) -> Result<Self, Error>
    where
        T: Read + Seek
    {
        let brush_id = data.read_u32::<LittleEndian>()?;
        let vert_chunk = Chunk::read(data)?;
        let vertices = Verts::read(data, vert_chunk.next)?;
        let mut triangles = Vec::new();

        while eof(data, next)? {
            let tri_chunk = Chunk::read(data)?;
            triangles.push(Tris::read(data, tri_chunk.next)?);
        }

        Ok(Self {
            brush_id,
            vertices,
            triangles,
        })
    }
}

#[derive(Debug, Default)]
pub struct Bone {
    pub vertex_id: u32,
    pub weight: f32,
}

impl Bone {
    pub fn read<T>(data: &mut T) -> Result<Self, Error>
    where
        T: Read
    {
        Ok(Self {
            vertex_id: data.read_u32::<LittleEndian>()?,
            weight: data.read_f32::<LittleEndian>()?,
        })
    }
}

#[derive(Debug, Default)]
pub struct Key {
    pub frame: u32,
    pub position: Vec3,
    pub scale: Vec3,
    pub rotation: Vec4,
}

impl Key {
    pub fn read<T>(data: &mut T, flags: u32) -> Result<Self, Error>
    where
        T: Read + Seek
    {
        let frame = data.read_u32::<LittleEndian>()?;

        let mut position = [0.0; 3];
        if flags & 1 != 0 {
            data.read_f32_into::<LittleEndian>(&mut position)?;
        }
        let mut scale = [0.0; 3];
        if flags & 2 != 0 {
            data.read_f32_into::<LittleEndian>(&mut scale)?;
        }
        let mut rotation = [0.0; 4];
        if flags & 4 != 0 {
            data.read_f32_into::<LittleEndian>(&mut rotation)?;
        }

        Ok(Self {
            frame,
            position,
            scale,
            rotation,
        })
    }
}

#[derive(Debug, Default)]
pub struct Animation {
    pub flags: u32,
    pub frames: u32,
    pub fps: f32,
}

impl Animation {
    pub fn read<T>(data: &mut T, _next: u64) -> Result<Self, Error>
    where
        T: Read + Seek
    {
        Ok(Self {
            flags: data.read_u32::<LittleEndian>()?,
            frames: data.read_u32::<LittleEndian>()?,
            fps: data.read_f32::<LittleEndian>()?,
        })
    }
}

#[derive(Debug, Default)]
pub struct Sequence {
    pub name: String,
    pub first_frame: u32,
    pub last_frame: u32,
    pub unused: u32,
}

impl Sequence {
    pub fn read<T>(data: &mut T, _next: u64) -> Result<Self, Error>
    where
        T: Read + Seek
    {
        Ok(Self {
            name: read_null_term_string(data),
            first_frame: data.read_u32::<LittleEndian>()?,
            last_frame: data.read_u32::<LittleEndian>()?,
            unused: data.read_u32::<LittleEndian>()?,
        })
    }
}

#[derive(Debug, Default)]
pub struct Node {
    pub name: String,
    pub position: Vec3,
    pub scale: Vec3,
    pub rotation: Vec4,
    pub mesh: Mesh,
    pub bones: Vec<Bone>,
    pub key_flags: u32,
    pub keys: Vec<Key>,
    pub children: Vec<Node>,
    pub animation: Animation,
    pub sequences: Vec<Sequence>,
}

impl Node {
    pub fn read<T>(data: &mut T, next: u64) -> Result<Self, Error>
    where
        T: Read + Seek
    {
        let name = read_null_term_string(data);
        let mut position = [0.0; 3];
        data.read_f32_into::<LittleEndian>(&mut position)?;
        let mut scale = [0.0; 3];
        data.read_f32_into::<LittleEndian>(&mut scale)?;
        let mut rotation = [0.0; 4];
        data.read_f32_into::<LittleEndian>(&mut rotation)?;

        let mut mesh = Mesh::default();
        let mut children = Vec::new();
        let mut bones = Vec::new();
        let mut animation = Animation::default();
        let mut sequences = Vec::new();
        let mut key_flags = 0;
        let mut keys = Vec::new();

        while eof(data, next)? {
            let chunk = Chunk::read(data)?;
            match chunk.tag.as_str() {
                "MESH" => mesh = Mesh::read(data, chunk.next)?,
                "BONE" => bones = Self::read_bones(data, chunk.next)?,
                "KEYS" => {
                    let kf = data.read_u32::<LittleEndian>()?;
                    key_flags |= kf;
                    keys.extend(Self::read_keys(data, chunk.next, kf)?);
                },
                "NODE" => children.push(Node::read(data, chunk.next)?),
                "ANIM" => animation = Animation::read(data, chunk.next)?,
                "SEQS" => sequences.push(Sequence::read(data, chunk.next)?),
                "PIVO" => {
                    let mut buf = vec![0; chunk.size as usize];
                    data.read_exact(&mut buf)?;
                }
                _ => return Err(Error::InvalidChunk(chunk).into()),
            }
        }

        Ok(Self {
            name,
            position,
            scale,
            rotation,
            mesh,
            bones,
            key_flags,
            keys,
            children,
            animation,
            sequences,
        })
    }

    pub fn read_bones<T>(data: &mut T, next: u64) -> Result<Vec<Bone>, Error>
    where
        T: Read + Seek
    {
        let mut bones = vec![];
        while eof(data, next)? {
            bones.push(Bone::read(data)?);
        }
        Ok(bones)
    }

    pub fn read_keys<T>(data: &mut T, next: u64, flags: u32) -> Result<Vec<Key>, Error>
    where
        T: Read + Seek
    {
        let mut keys = vec![];
        while eof(data, next)? {
            keys.push(Key::read(data, flags)?);
        }
        Ok(keys)
    }
}

#[derive(Debug)]
pub struct B3D {
    pub version: u32,
    pub textures: Vec<Texture>,
    pub brushes: Vec<Brush>,
    pub node: Node,
}

impl B3D {
    pub fn read(data: &[u8]) -> Result<Self, Error> {
        let mut cursor = Cursor::new(data);

        let main_chunk = Chunk::read(&mut cursor)?;
        if main_chunk.tag != "BB3D" {
            return Err(Error::InvalidChunk(main_chunk).into());
        }
        let version = cursor.read_u32::<LittleEndian>()?;
        let mut textures = Vec::new();
        let mut brushes = Vec::new();
        let mut node = Node::default();

        while eof(&mut cursor, main_chunk.next)? {
            let chunk = Chunk::read(&mut cursor)?;
            match chunk.tag.as_str() {
                "TEXS" => textures = Self::read_textures(&mut cursor, chunk.next)?,
                "BRUS" => brushes = Self::read_brushes(&mut cursor, chunk.next)?,
                "NODE" => node = Node::read(&mut cursor, chunk.next)?,
                "PIVO" => {
                    let mut buf = vec![0; chunk.size as usize];
                    cursor.read_exact(&mut buf)?;
                }
                _ => return Err(Error::InvalidChunk(chunk).into()),
            }
        }

        Ok(Self {
            version,
            textures,
            brushes,
            node,
        })
    }

    pub fn read_textures<T>(data: &mut T, next: u64) -> Result<Vec<Texture>, Error>
    where
        T: Read + Seek
    {
        let mut textures = vec![];
        while eof(data, next)? {
            textures.push(Texture::read(data)?);
        }
        Ok(textures)
    }

    pub fn read_brushes<T>(data: &mut T, next: u64) -> Result<Vec<Brush>, Error>
    where
        T: Read + Seek
    {
        let mut brushes = vec![];
        let n_texs = data.read_u32::<LittleEndian>()?;
        while eof(data, next)? {
            brushes.push(Brush::read(data, n_texs as usize)?);
        }
        Ok(brushes)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers: binary writing for synthetic data ───────────────────────
    fn write_chunk(buf: &mut Vec<u8>, tag: &[u8; 4], size: u32) {
        buf.extend_from_slice(tag);
        buf.write_u32::<LittleEndian>(size).unwrap();
    }

    fn write_f32_3(buf: &mut Vec<u8>, v: [f32; 3]) {
        for x in v { buf.write_f32::<LittleEndian>(x).unwrap(); }
    }

    fn write_f32_4(buf: &mut Vec<u8>, v: [f32; 4]) {
        for x in v { buf.write_f32::<LittleEndian>(x).unwrap(); }
    }

    // ── Helper: count all nodes in the tree ──────────────────────────────
    fn count_nodes(node: &Node) -> usize {
        1 + node.children.iter().map(count_nodes).sum::<usize>()
    }

    // ── Helper: find a node by name in the tree ──────────────────────────
    fn find_node<'a>(node: &'a Node, name: &str) -> Option<&'a Node> {
        if node.name == name { return Some(node); }
        for child in &node.children {
            if let Some(found) = find_node(child, name) { return Some(found); }
        }
        None
    }

    // ── Helper: collect all node names in traversal order ────────────────
    fn collect_names(node: &Node) -> Vec<String> {
        let mut names = vec![node.name.clone()];
        for child in &node.children {
            names.extend(collect_names(child));
        }
        names
    }

    // ===================================================================
    // Monkey.b3d integration tests
    // ===================================================================
    #[test]
    fn test_parse_monkey_version() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        assert_eq!(b3d.version, 1);
    }

    #[test]
    fn test_parse_monkey_node_count() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        // ROOT + 18 joint nodes = 19 total
        assert_eq!(count_nodes(&b3d.node), 19);
    }

    #[test]
    fn test_parse_monkey_node_names() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        let names = collect_names(&b3d.node);
        let expected: Vec<&str> = vec![
            "ROOT",
            "joint1", "joint2", "joint13", "joint14",
            "joint3", "joint15", "joint16",
            "joint4",
            "joint5", "joint17",
            "joint6", "joint18",
            "joint7", "joint8", "joint9", "joint10", "joint11", "joint12",
        ];
        assert_eq!(names.len(), expected.len());
        for (got, exp) in names.iter().zip(expected.iter()) {
            assert_eq!(got, exp, "node name mismatch");
        }
    }

    #[test]
    fn test_parse_monkey_root_mesh() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        // Only ROOT has a mesh
        assert!(!b3d.node.mesh.vertices.vertices.is_empty(),
            "ROOT should have mesh vertices");
        assert_eq!(b3d.node.mesh.vertices.vertices.len(), 295);
        // Check vertex flags: 0 means no normals/colors stored per vertex
        let verts = &b3d.node.mesh.vertices;
        assert_eq!(verts.flags, 0, "expected no normal/color flags");
    }

    #[test]
    fn test_parse_monkey_mesh_triangles() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        // ROOT mesh has triangles (expected ~500+ depending on monkey detail)
        let tri_count: usize = b3d.node.mesh.triangles.iter()
            .map(|t| t.indices.len()).sum();
        assert!(tri_count > 100, "expected >100 triangles, got {tri_count}");
    }

    #[test]
    fn test_parse_monkey_joint1_bones() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        let j1 = find_node(&b3d.node, "joint1").unwrap();
        assert_eq!(j1.bones.len(), 37, "joint1 should have 37 bones");
        // Check bone 0
        assert_eq!(j1.bones[0].vertex_id, 73);
        assert!((j1.bones[0].weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_parse_monkey_joint4_bones() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        let j4 = find_node(&b3d.node, "joint4").unwrap();
        assert_eq!(j4.bones.len(), 84, "joint4 should have 84 bones");
    }

    #[test]
    fn test_parse_monkey_joint12_bones() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        let j12 = find_node(&b3d.node, "joint12").unwrap();
        assert_eq!(j12.bones.len(), 6, "joint12 should have 6 bones");
    }

    // ── Verify KEYS on every joint ───────────────────────────────────────
    // Each joint node should have 34 keys with both position and rotation
    // data (from separate KEYS chunks merged by the fix).
    #[test]
    fn test_parse_monkey_all_joints_have_keys() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        let joint_names = [
            "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
            "joint7", "joint8", "joint9", "joint10", "joint11", "joint12",
            "joint13", "joint14", "joint15", "joint16", "joint17", "joint18",
        ];
        for name in &joint_names {
            let node = find_node(&b3d.node, name)
                .unwrap_or_else(|| panic!("{name} not found"));
            assert!(node.key_flags > 0,
                "{name}: key_flags should be non-zero (has position/rotation)");
            // Each bone should have position (1) and rotation (4) keys → flags=5
            assert!(node.key_flags & 1 != 0, "{name}: missing position keys (flags&1)");
            assert!(node.key_flags & 4 != 0, "{name}: missing rotation keys (flags&4)");
            assert_eq!(node.keys.len(), 34,
                "{name}: expected 34 keys, got {}", node.keys.len());
            // First key should have non-zero position data
            let first = &node.keys[0];
            assert!(first.frame == 0 || first.frame == 1,
                "{name}: first key frame should be 0 or 1");
            // Verify position data varies across keys (not all zeros)
            let last = &node.keys[node.keys.len() - 1];
            let pos_diff =
                (last.position[0] - first.position[0]).abs() +
                (last.position[1] - first.position[1]).abs() +
                (last.position[2] - first.position[2]).abs();
            assert!(pos_diff > 0.001 || last.rotation != first.rotation,
                "{name}: keys should have animation data");
        }
    }

    #[test]
    fn test_parse_monkey_root_has_no_keys() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        assert_eq!(b3d.node.keys.len(), 0, "ROOT should have no keys");
        assert_eq!(b3d.node.key_flags, 0, "ROOT should have no key_flags");
    }

    // ── Verify position / rotation values on a known joint ───────────────
    #[test]
    fn test_parse_monkey_joint1_identity_pose() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        let j1 = find_node(&b3d.node, "joint1").unwrap();
        // joint1 is at (0.0333, 17.5667, -6.0212) with identity rotation
        assert!((j1.position[0] - 0.03333396).abs() < 0.001);
        assert!((j1.position[1] - 17.56666756).abs() < 0.001);
        assert!((j1.position[2] - (-6.02116012)).abs() < 0.001);
        // Expect identity rotation [w, x, y, z] ≈ [1, 0, 0, 0]
        assert!((j1.rotation[0] - 1.0).abs() < 0.001);
        assert!((j1.rotation[1]).abs() < 0.001);
        assert!((j1.rotation[2]).abs() < 0.001);
        assert!((j1.rotation[3]).abs() < 0.001);
    }

    #[test]
    fn test_parse_monkey_joint2_pose() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        let j2 = find_node(&b3d.node, "joint2").unwrap();
        // joint2 at (-9.5327, 5.2530, 4.6003) with non-identity rotation
        assert!((j2.position[0] - (-9.53267670)).abs() < 0.001);
        assert!((j2.position[1] - 5.25301361).abs() < 0.001);
        assert!((j2.position[2] - 4.60026455).abs() < 0.001);
    }

    // ── Verify child/parent hierarchy ────────────────────────────────────
    #[test]
    fn test_parse_monkey_parent_child_relationships() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        // ROOT has 1 direct child (joint1)
        assert_eq!(b3d.node.children.len(), 1,
            "ROOT should have 1 direct child (joint1)");
        // joint1 has 6 children (joint2..joint7)
        let j1 = find_node(&b3d.node, "joint1").unwrap();
        assert_eq!(j1.children.len(), 6,
            "joint1 should have 6 direct children, got {}", j1.children.len());
        // joint2 is child of joint1
        assert!(j1.children.iter().any(|c| c.name == "joint2"),
            "joint1 should have joint2 as child");
        // joint13 is child of joint2
        let j2 = find_node(&b3d.node, "joint2").unwrap();
        assert!(j2.children.iter().any(|c| c.name == "joint13"),
            "joint2 should have joint13 as child");
        // joint14 is child of joint13
        let j13 = find_node(&b3d.node, "joint13").unwrap();
        assert!(j13.children.iter().any(|c| c.name == "joint14"),
            "joint13 should have joint14 as child");
        // Leaf nodes have no children
        let j14 = find_node(&b3d.node, "joint14").unwrap();
        assert_eq!(j14.children.len(), 0, "joint14 should have no children");
    }

    #[test]
    fn test_parse_monkey_brushes_textures() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        // monkey.b3d has 1 brush ("monkeyskin.bmp")
        assert_eq!(b3d.brushes.len(), 1,
            "expected 1 brush, got {}", b3d.brushes.len());
        assert_eq!(b3d.brushes[0].texture_id.len(), 1,
            "brush should reference 1 texture");
        assert_eq!(b3d.textures.len(), 1,
            "expected 1 texture, got {}", b3d.textures.len());
        assert!(b3d.textures[0].file.contains("monkeyskin"),
            "texture file should contain 'monkeyskin'");
    }

    // ── Verify scale is identity for all joints ──────────────────────────
    #[test]
    fn test_parse_monkey_all_scales_identity() {
        let data = include_bytes!("../tests/data/monkey.b3d");
        let b3d = B3D::read(data).unwrap();
        let names = collect_names(&b3d.node);
        for name in &names {
            let node = find_node(&b3d.node, name).unwrap();
            assert!((node.scale[0] - 1.0).abs() < 0.001,
                "{name}: scale.x != 1");
            assert!((node.scale[1] - 1.0).abs() < 0.001,
                "{name}: scale.y != 1");
            assert!((node.scale[2] - 1.0).abs() < 0.001,
                "{name}: scale.z != 1");
        }
    }

    // ===================================================================
    // Error handling tests
    // ===================================================================
    #[test]
    fn test_invalid_magic() {
        let bad = b"BBXD\x00\x00\x00\x00\x01\x00\x00\x00";
        let result = B3D::read(bad);
        assert!(result.is_err(), "should reject invalid magic");
    }

    #[test]
    fn test_empty_data() {
        let result = B3D::read(b"");
        assert!(result.is_err(), "should reject empty data");
    }

    #[test]
    fn test_truncated_data() {
        // Valid header but no body
        let bad = b"BB3D\x14\x00\x00\x00\x01\x00\x00\x00";
        let result = B3D::read(bad);
        assert!(result.is_err(), "should reject truncated data");
    }

    #[test]
    fn test_too_short_header() {
        let result = B3D::read(b"BB3");
        assert!(result.is_err(), "should reject too-short header");
    }

    #[test]
    fn test_unknown_chunk_type() {
        // BB3D header with an unknown chunk tag "XXXX"
        let data = b"BB3D\x1c\x00\x00\x00\x01\x00\x00\x00XXXX\x08\x00\x00\x00\x00\x00\x00\x00";
        let result = B3D::read(data);
        assert!(result.is_err(), "should reject unknown chunk");
    }

    // ===================================================================
    // KEYS merge test (critical fix verification)
    // ===================================================================
    /// Construct a synthetic B3D with a single node that has TWO KEYS chunks:
    /// first with flags=1 (position), second with flags=4 (rotation).
    /// After parsing, the node should have both position and rotation data
    /// in its keys, and key_flags should be 5 (1|4).
    #[test]
    fn test_keys_merge_multiple_chunks() {
        let mut data = Vec::new();

        // ── BB3D header ──────────────────────────────────────────────
        write_chunk(&mut data, b"BB3D", 0); // size placeholder
        data.write_u32::<LittleEndian>(1).unwrap(); // version = 1

        // ── NODE chunk ────────────────────────────────────────────────
        write_chunk(&mut data, b"NODE", 0); // size placeholder
        let node_start = data.len();

        // Node name + TRS
        data.extend_from_slice(b"test_joint\x00"); // null-terminated name
        write_f32_3(&mut data, [1.0, 2.0, 3.0]); // position
        write_f32_3(&mut data, [1.0, 1.0, 1.0]); // scale
        write_f32_4(&mut data, [1.0, 0.0, 0.0, 0.0]); // rotation (identity)

        // ── First KEYS chunk (flags=1: position only) ─────────────────
        let keys1_start = data.len();
        write_chunk(&mut data, b"KEYS", 0);
        let keys1_data_start = data.len();
        data.write_u32::<LittleEndian>(1).unwrap(); // flags = 1 (position)
        // Two keyframes with position data
        data.write_u32::<LittleEndian>(0).unwrap(); // frame 0
        write_f32_3(&mut data, [10.0, 20.0, 30.0]); // position
        data.write_u32::<LittleEndian>(1).unwrap(); // frame 1
        write_f32_3(&mut data, [11.0, 21.0, 31.0]); // position
        // Fill in KEYS chunk size
        let keys1_size = (data.len() - keys1_data_start) as u32;
        let keys1_size_field = keys1_start + 4;
        data[keys1_size_field..keys1_size_field + 4]
            .copy_from_slice(&keys1_size.to_le_bytes());

        // ── Second KEYS chunk (flags=4: rotation only) ────────────────
        let keys2_start = data.len();
        write_chunk(&mut data, b"KEYS", 0);
        let keys2_data_start = data.len();
        data.write_u32::<LittleEndian>(4).unwrap(); // flags = 4 (rotation)
        // Two keyframes with rotation data (same frames as above)
        data.write_u32::<LittleEndian>(0).unwrap(); // frame 0
        write_f32_4(&mut data, [0.7071, 0.7071, 0.0, 0.0]); // rotation
        data.write_u32::<LittleEndian>(1).unwrap(); // frame 1
        write_f32_4(&mut data, [0.0, 1.0, 0.0, 0.0]); // rotation
        // Fill in KEYS chunk size
        let keys2_size = (data.len() - keys2_data_start) as u32;
        let keys2_size_field = keys2_start + 4;
        data[keys2_size_field..keys2_size_field + 4]
            .copy_from_slice(&keys2_size.to_le_bytes());

        // ── Fill in NODE chunk size ───────────────────────────────────
        let node_size = (data.len() - node_start) as u32;
        let node_size_field = node_start - 4;
        data[node_size_field..node_size_field + 4]
            .copy_from_slice(&node_size.to_le_bytes());

        // ── Fill in BB3D chunk size ───────────────────────────────────
        let bb3d_size = (data.len() - 8) as u32; // 8 = tag + size
        let bb3d_size_field = 4;
        data[bb3d_size_field..bb3d_size_field + 4]
            .copy_from_slice(&bb3d_size.to_le_bytes());

        // ── Parse and verify ──────────────────────────────────────────
        let b3d = B3D::read(&data).unwrap();
        assert_eq!(b3d.version, 1);
        assert_eq!(b3d.node.name, "test_joint");

        // THE CRITICAL ASSERTION: key_flags should be 5 (1|4)
        assert_eq!(b3d.node.key_flags, 5,
            "key_flags should be 1|4=5 after merging two KEYS chunks, got {}",
            b3d.node.key_flags);

        // Should have 4 keys: 2 from position KEYS + 2 from rotation KEYS.
        // Each KEYS chunk contributes its keyframes independently.
        assert_eq!(b3d.node.keys.len(), 4,
            "should have 4 keys (2 pos + 2 rot), got {}", b3d.node.keys.len());

        // Keys[0..1] are from position KEYS chunk (rotation fields are zero)
        assert_eq!(b3d.node.keys[0].frame, 0);
        assert!((b3d.node.keys[0].position[0] - 10.0).abs() < 0.001);
        assert!((b3d.node.keys[0].position[1] - 20.0).abs() < 0.001);
        assert!((b3d.node.keys[0].position[2] - 30.0).abs() < 0.001);
        assert_eq!(b3d.node.keys[0].rotation, [0.0, 0.0, 0.0, 0.0],
            "position KEYS chunk should not have rotation data");
        assert_eq!(b3d.node.keys[1].frame, 1);
        assert!((b3d.node.keys[1].position[0] - 11.0).abs() < 0.001);

        // Keys[2..3] are from rotation KEYS chunk (position fields are zero)
        assert_eq!(b3d.node.keys[2].frame, 0);
        assert!((b3d.node.keys[2].rotation[0] - 0.7071).abs() < 0.001);
        assert!((b3d.node.keys[2].rotation[1] - 0.7071).abs() < 0.001);
        assert_eq!(b3d.node.keys[2].position, [0.0, 0.0, 0.0],
            "rotation KEYS chunk should not have position data");
        assert_eq!(b3d.node.keys[3].frame, 1);
        assert!((b3d.node.keys[3].rotation[1] - 1.0).abs() < 0.001);
    }

    // ===================================================================
    // Utility tests
    // ===================================================================
    #[test]
    fn test_chunk_read_valid() {
        let mut buf = Cursor::new(b"TEXS\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
        let chunk = Chunk::read(&mut buf).unwrap();
        assert_eq!(chunk.tag, "TEXS");
        assert_eq!(chunk.size, 16);
    }

    #[test]
    fn test_chunk_display() {
        let mut buf = Cursor::new(b"TEXS\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
        let chunk = Chunk::read(&mut buf).unwrap();
        let s = format!("{chunk}");
        assert!(s.contains("TEXS"));
        assert!(s.contains("Size: 16"));
    }

    #[test]
    fn test_read_null_term_string() {
        let mut buf = Cursor::new(b"hello\x00world");
        let s = read_null_term_string(&mut buf);
        assert_eq!(s, "hello");
        // cursor should now be at 'w'
        let rest: Vec<u8> = {
            let mut r = vec![];
            buf.read_to_end(&mut r).unwrap();
            r
        };
        assert_eq!(rest, b"world");
    }

    #[test]
    fn test_eof_before_end() {
        let mut buf = Cursor::new(b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00");
        let result = eof(&mut buf, 100).unwrap();
        assert!(result, "stream_pos < next, should be true");
    }

    #[test]
    fn test_eof_at_end() {
        let mut buf = Cursor::new(b"\x00\x00\x00\x00");
        buf.seek(std::io::SeekFrom::End(0)).unwrap();
        let result = eof(&mut buf, 4).unwrap();
        assert!(!result, "stream_pos == next, should be false");
    }

    // ===================================================================
    // Individual struct reader tests with synthetic data
    // ===================================================================
    #[test]
    fn test_read_bone() {
        let mut buf = Cursor::new(b"\x2a\x00\x00\x00\x00\x00\x80\x3f");
        let bone = Bone::read(&mut buf).unwrap();
        assert_eq!(bone.vertex_id, 42);
        assert!((bone.weight - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_read_key_position_only() {
        let mut buf = Cursor::new(b"\x05\x00\x00\x00\x00\x00\x00\x40\x00\x00\x40\x40\x00\x00\x80\x40");
        // flags=1: frame=5, position=[2.0, 3.0, 4.0]
        let key = Key::read(&mut buf, 1).unwrap();
        assert_eq!(key.frame, 5);
        assert!((key.position[0] - 2.0).abs() < 0.001);
        assert!((key.position[1] - 3.0).abs() < 0.001);
        assert!((key.position[2] - 4.0).abs() < 0.001);
        // scale and rotation should be zero (not read)
        assert_eq!(key.scale, [0.0, 0.0, 0.0]);
        assert_eq!(key.rotation, [0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_read_key_rotation_only() {
        let mut buf = Cursor::new(b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x3f");
        // flags=4: frame=0, no position, no scale, rotation=[0,0,0,1]
        let key = Key::read(&mut buf, 4).unwrap();
        assert_eq!(key.frame, 0);
        assert_eq!(key.position, [0.0, 0.0, 0.0]);
        assert_eq!(key.scale, [0.0, 0.0, 0.0]);
        assert!((key.rotation[3] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_read_texture() {
        let mut buf = Vec::new();
        // file name
        buf.extend_from_slice(b"test.tex\x00");
        // flags=1, blend=2
        buf.write_u32::<LittleEndian>(1).unwrap();
        buf.write_u32::<LittleEndian>(2).unwrap();
        // position [0.5, 1.5] (2 × f32)
        buf.write_f32::<LittleEndian>(0.5).unwrap();
        buf.write_f32::<LittleEndian>(1.5).unwrap();
        // scale [2.0, 3.0] (2 × f32)
        buf.write_f32::<LittleEndian>(2.0).unwrap();
        buf.write_f32::<LittleEndian>(3.0).unwrap();
        // rotation 45° (1 × f32)
        buf.write_f32::<LittleEndian>(45.0_f32.to_radians()).unwrap();
        let tex = Texture::read(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(tex.file, "test.tex");
        assert_eq!(tex.flags, 1);
        assert_eq!(tex.blend, 2);
        assert!((tex.position[0] - 0.5).abs() < 0.001);
        assert!((tex.scale[0] - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_read_animation() {
        let mut buf = Cursor::new(b"\x01\x00\x00\x00\x64\x00\x00\x00\x00\x00\x00\x00");
        // flags=1, frames=100, fps=0? Actually let me check:
        // FPS is f32. Let me verify: 0x00000000 as f32 = 0.0
        let anim = Animation::read(&mut buf, 0).unwrap();
        assert_eq!(anim.flags, 1);
        assert_eq!(anim.frames, 100);
    }

    // ===================================================================
    // Read_keys helper test
    // ===================================================================
    #[test]
    fn test_read_keys_with_position_flags() {
        let data = b"\x00\x00\x00\x00\x00\x00\x00\x40\x00\x00\x00\x40\x00\x00\x00\x40\x01\x00\x00\x00\x00\x00\x80\x40\x00\x00\x40\x40\x00\x00\x00\x40";
        // Two keys with flags=1 (position only, 16 bytes each = 4(frame)+12(pos))
        // key0: frame=0, pos=[2.0, 2.0, 2.0]
        // key1: frame=1, pos=[4.0, 3.0, 2.0]
        let keys = Node::read_keys(&mut Cursor::new(data), data.len() as u64, 1).unwrap();
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].frame, 0);
        assert!((keys[0].position[0] - 2.0).abs() < 0.001);
        assert_eq!(keys[1].frame, 1);
        assert!((keys[1].position[0] - 4.0).abs() < 0.001);
    }

    // ===================================================================
    // Empty node parsing
    // ===================================================================
    #[test]
    fn test_minimal_node() {
        let mut data = Vec::new();
        // Write just the TRS data that Node::read expects at offset 0
        data.extend_from_slice(b"test\x00");
        write_f32_3(&mut data, [0.0, 0.0, 0.0]); // position
        write_f32_3(&mut data, [1.0, 1.0, 1.0]); // scale
        write_f32_4(&mut data, [1.0, 0.0, 0.0, 0.0]); // rotation
        // No child chunks — node should parse with defaults
        let mut cursor = Cursor::new(&data);
        let node = Node::read(&mut cursor, data.len() as u64).unwrap();
        assert_eq!(node.name, "test");
        assert_eq!(node.children.len(), 0);
        assert_eq!(node.bones.len(), 0);
        assert_eq!(node.keys.len(), 0);
    }
}
