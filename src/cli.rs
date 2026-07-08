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

use std::path::PathBuf;

/// Expand a leading `~` to the user's home directory.
fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~") {
        if let Ok(home) = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
        {
            if rest.is_empty() || rest.starts_with('/') || rest.starts_with('\\') {
                let mut p = PathBuf::from(home);
                p.push(rest.trim_start_matches('/').trim_start_matches('\\'));
                return p;
            }
        }
    }
    PathBuf::from(path)
}

/// Override material PBR parameters (metallic, roughness).
#[derive(Debug, Clone, Copy)]
pub struct MaterialParams {
    pub metallic: f32,
    pub roughness: f32,
}

/// Parsed command-line arguments.
#[derive(Debug)]
pub struct Args {
    /// Input paths (files or directories).
    pub inputs: Vec<PathBuf>,
    /// Output directory (default: current dir).
    pub out_dir: PathBuf,
    /// Context / game directory for texture lookups.
    pub context_dir: Option<PathBuf>,
    /// Whether to write a single .glb (otherwise .gltf + .bin + textures).
    pub glb: bool,
    /// Override material PBR parameters (metallic, roughness).
    pub material_params: Option<MaterialParams>,
    /// Override baseColorFactor for non-textured materials (r,g,b[,a]).
    pub color_override: Option<[f32; 4]>,
}

/// Parse a color string "r,g,b[,a]" (each 0.0–1.0).
fn parse_color(s: &str) -> Result<[f32; 4], String> {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() < 3 || parts.len() > 4 {
        return Err(format!("invalid color format '{s}'. Use 'r,g,b' or 'r,g,b,a' (e.g. 1.0,0.5,0.0)"));
    }
    let r: f32 = parts[0].trim().parse().map_err(|_| format!("invalid red: {}", parts[0]))?;
    let g: f32 = parts[1].trim().parse().map_err(|_| format!("invalid green: {}", parts[1]))?;
    let b: f32 = parts[2].trim().parse().map_err(|_| format!("invalid blue: {}", parts[2]))?;
    let a: f32 = if parts.len() == 4 {
        parts[3].trim().parse().map_err(|_| format!("invalid alpha: {}", parts[3]))?
    } else {
        1.0
    };
    Ok([r, g, b, a])
}

fn parse_material(s: &str) -> Result<MaterialParams, String> {
    // Parse format like "0.0m0.9r" or "0.5,0.3"
    if let Some((metallic_str, rest)) = s.split_once('m') {
        if let Some(roughness_str) = rest.strip_suffix('r') {
            let metallic: f32 = metallic_str.parse()
                .map_err(|_| format!("invalid metallic value: {metallic_str}"))?;
            let roughness: f32 = roughness_str.parse()
                .map_err(|_| format!("invalid roughness value: {roughness_str}"))?;
            return Ok(MaterialParams { metallic, roughness });
        }
    }
    // Fallback: comma-separated "metallic,roughness"
    if let Some((m, r)) = s.split_once(',') {
        let metallic: f32 = m.trim().parse()
            .map_err(|_| format!("invalid metallic value: {m}"))?;
        let roughness: f32 = r.trim().parse()
            .map_err(|_| format!("invalid roughness value: {r}"))?;
        return Ok(MaterialParams { metallic, roughness });
    }
    Err(format!("invalid material format '{s}'. Use '<metallic>m<roughness>r' (e.g. 0.0m0.9r) or '<metallic>,<roughness>' (e.g. 0.0,0.9)"))
}

const USAGE: &str = "\
b3d2glb — convert Blitz3D .b3d models to glTF 2.0

USAGE:
  b3d2glb [OPTIONS] input...

ARGS:
  input...   One or more .b3d files or directories containing .b3d files.

OPTIONS:
  -o, --out DIR           Output directory (default: current directory)
  -c, --context DIR       Context / game root directory (texture lookup root)
  -b, --glb               Write binary .glb instead of separate .gltf + .bin + textures
  -m, --material PARAMS   Override material params (e.g. 0.0m0.9r or 0.0,0.9)
  -C, --color R,G,B[,A]   Base color for non-textured materials (default: 0.8,0.8,0.8,1)
  -h, --help              Display this help and exit

EXAMPLES:
  b3d2glb -o ./out -c /path/to/game model.b3d
  b3d2glb --glb -o ./out /path/to/game/gfx
  b3d2glb -b model.b3d
  b3d2glb -b -m 0.0m0.9r model.b3d
  b3d2glb -b -C 0.8,0.8,0.8 model.b3d
";

/// Parse command-line arguments or print help and exit.
pub fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();
    let mut args = Args {
        inputs: Vec::new(),
        out_dir: PathBuf::from("."),
        context_dir: None,
        glb: false,
        material_params: None,
        color_override: None,
    };

    let mut i = 1;
    while i < raw.len() {
        match raw[i].as_str() {
            "-h" | "--help" | "-?" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "-o" | "--out" => {
                i += 1;
                if i >= raw.len() {
                    return Err("-o/--out requires a value".into());
                }
                args.out_dir = expand_tilde(&raw[i]);
            }
            "-c" | "--context" => {
                i += 1;
                if i >= raw.len() {
                    return Err("-c/--context requires a value".into());
                }
                args.context_dir = Some(expand_tilde(&raw[i]));
            }
            "-m" | "--material" => {
                i += 1;
                if i >= raw.len() {
                    return Err("-m/--material requires a value".into());
                }
                args.material_params = Some(parse_material(&raw[i])?);
            }
            "-C" | "--color" => {
                i += 1;
                if i >= raw.len() {
                    return Err("-C/--color requires a value".into());
                }
                args.color_override = Some(parse_color(&raw[i])?);
            }
            "-b" | "--glb" => {
                args.glb = true;
            }
            s if s.starts_with('-') => {
                return Err(format!("unknown option: {s}"));
            }
            _ => {
                args.inputs.push(expand_tilde(&raw[i]));
            }
        }
        i += 1;
    }

    if args.inputs.is_empty() {
        return Err("no input files or directories specified".into());
    }

    Ok(args)
}
