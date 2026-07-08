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

use std::fs;
use std::env;
use b3d2glb::b3d_parser::B3D;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 { eprintln!("usage: dump <file.b3d>"); return; }
    let data = fs::read(&args[1]).unwrap();
    let b3d = B3D::read(&data).unwrap();
    let vcount = b3d.node.mesh.vertices.vertices.len();

    println!("=== ROOT NODE ===");
    println!("name: {}", b3d.node.name);
    println!("position: {:?}", b3d.node.position);
    println!("scale: {:?}", b3d.node.scale);
    println!("rotation: {:?}", b3d.node.rotation);
    println!();

    dump_node(&b3d.node, 0, vcount);
}

fn dump_node(node: &b3d2glb::b3d_parser::Node, depth: usize, vcount: usize) {
    let indent = "  ".repeat(depth);
    let has_mesh = !node.mesh.vertices.vertices.is_empty();
    if has_mesh {
        println!("{}+ MESH \"{}\" pos={:?} sc={:?} rot={:?} bones={} verts={}", indent,
            node.name, node.position, node.scale, node.rotation,
            node.bones.len(), node.mesh.vertices.vertices.len());
    } else {
        println!("{}+ NODE \"{}\" pos=({:.4},{:.4},{:.4}) sc=({:.4},{:.4},{:.4}) rot=({:.4},{:.4},{:.4},{:.4}) bones={} keys={}", indent,
            node.name,
            node.position[0], node.position[1], node.position[2],
            node.scale[0], node.scale[1], node.scale[2],
            node.rotation[0], node.rotation[1], node.rotation[2], node.rotation[3],
            node.bones.len(), node.keys.len());
    }
    for (i, b) in node.bones.iter().enumerate().take(8) {
        println!("{}  bone[{}]: v={} w={:.2}", indent, i, b.vertex_id, b.weight);
    }
    if node.bones.len() > 8 { println!("{}  ... ({} more)", indent, node.bones.len() - 8); }
    for child in &node.children { dump_node(child, depth + 1, vcount); }
}
