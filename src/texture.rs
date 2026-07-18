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
use std::path::{Path, PathBuf};

/// Known texture extensions in priority order (used as fallback).
const TEXTURE_EXTENSIONS: &[&str] = &["bmp", "jpg", "jpeg", "png", "tga"];

/// Try to find a texture at `base_path` by trying the original extension first,
/// then falling back to all known extensions.
fn try_extensions(base_path: &Path, orig_ext: Option<&str>) -> Option<PathBuf> {
    // 1. Try the original path as-is (includes its own extension).
    if base_path.exists() {
        return Some(base_path.to_path_buf());
    }

    // 2. Try original extension first (if known).
    if let Some(ext) = orig_ext {
        let ext_lower = ext.to_lowercase();
        if TEXTURE_EXTENSIONS.contains(&ext_lower.as_str()) {
            let p = base_path.with_extension(&ext_lower);
            if p.exists() {
                return Some(p);
            }
        }
    }

    // 3. Try all known extensions.
    for ext in TEXTURE_EXTENSIONS {
        // Skip if we already tried it above.
        if orig_ext.is_some_and(|e| e.eq_ignore_ascii_case(ext)) {
            continue;
        }
        let p = base_path.with_extension(ext);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

/// Find a texture file by its raw B3D path.
///
/// For each strategy, the **original extension** from the B3D path is tried
/// first (if present and known).  Fallback extensions follow in order:
/// `bmp`, `jpg`, `jpeg`, `png`, `tga`.
///
/// Search strategies (first match wins):
/// 1. `game_dir / raw_path` — preserves directory structure from B3D
/// 2. `game_dir / filename` — filename only, no directory
/// 3. `game_dir / lowercase_filename`
/// 4. Legacy Stranded II paths (`mods/Stranded II/gfx/` and `gfx/`)
pub fn find_texture(raw_path: &str, game_dir: &Path) -> Option<PathBuf> {
    // Normalize B3D Windows backslashes to forward slashes so Path works
    // on all platforms.
    let clean = raw_path
        .trim_start_matches(".\\")
        .trim_start_matches("./")
        .replace('\\', "/");
    let tex_path = Path::new(&clean);

    // Extract the original extension so we can prioritise it.
    let orig_ext = tex_path.extension().and_then(|e| e.to_str());

    // Strategy 1: game_dir / full original path (preserves directory structure)
    let full = game_dir.join(tex_path);
    if let Some(p) = try_extensions(&full, orig_ext) {
        return Some(p);
    }

    // Strategy 2: game_dir / filename only
    if let Some(file_name) = tex_path.file_name().and_then(|s| s.to_str()) {
        let base = game_dir.join(file_name);
        if let Some(p) = try_extensions(&base, orig_ext) {
            return Some(p);
        }
    }

    // Strategy 3: lowercase filename in game_dir
    if let Some(stem) = tex_path.file_stem().and_then(|s| s.to_str()) {
        let lower = stem.to_lowercase();
        let lower_path = game_dir.join(&lower);
        if let Some(p) = try_extensions(&lower_path, orig_ext) {
            return Some(p);
        }
    }

    // Strategy 4: legacy Stranded II paths
    if let Some(stem) = tex_path.file_stem().and_then(|s| s.to_str()) {
        for dir in &[game_dir.join("mods/Stranded II/gfx"), game_dir.join("gfx")] {
            for fname in &[stem, &stem.to_lowercase()] {
                let legacy_path = dir.join(fname);
                if let Some(p) = try_extensions(&legacy_path, orig_ext) {
                    return Some(p);
                }
            }
        }
    }

    None
}

/// Load a B3D texture from its raw B3D path, convert to PNG, cache to disk.
///
/// The cache key is the file stem (name without extension or directory).
/// Returns `None` if the texture cannot be found or decoded.
pub fn load_texture(raw_path: &str, game_dir: &Path, tex_cache: &Path) -> Option<Vec<u8>> {
    let stem = texture_stem(raw_path);
    let png_path = tex_cache.join(format!("{stem}.png"));

    // Return cached version if it exists.
    if png_path.exists() {
        return fs::read(&png_path).ok();
    }

    // Find and convert.
    let src = find_texture(raw_path, game_dir)?;
    let img = image::open(&src).ok()?;
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).ok()?;
    let bytes = buf.into_inner();

    // Cache to disk for subsequent runs.
    if let Err(e) = fs::write(&png_path, &bytes) {
        eprintln!("warning: failed to cache texture '{stem}': {e}");
    }
    Some(bytes)
}

/// Decode PNG bytes and return `true` if any pixel has alpha < 255.
///
/// This is a fallback when B3D metadata doesn't indicate transparency:
/// we check the actual image data for non-opaque pixels.
pub fn png_has_alpha(data: &[u8]) -> bool {
    use image::GenericImageView;
    let img = match image::load_from_memory(data) {
        Ok(img) => img,
        Err(_) => return false,
    };
    for (_x, _y, px) in img.pixels() {
        if px[3] < 255 {
            return true;
        }
    }
    false
}

/// Decode PNG bytes and return `true` if any pixel has semi-transparent alpha
/// (between 1 and 254 inclusive). Fully transparent (0) or fully opaque (255)
/// pixels don't count.
///
/// This is used to decide between `"BLEND"` (smooth transparency, needs
/// semi-transparent pixels) and `"MASK"` (hard cutoff, all transparent pixels
/// are fully transparent).
pub fn png_has_semi_transparent(data: &[u8]) -> bool {
    use image::GenericImageView;
    let img = match image::load_from_memory(data) {
        Ok(img) => img,
        Err(_) => return false,
    };
    for (_x, _y, px) in img.pixels() {
        let a = px[3];
        if a > 0 && a < 255 {
            return true;
        }
    }
    false
}

/// Extract the file stem (name without extension or directory) from a raw
/// B3D texture path.  E.g. `"gfx\\monkeyskin.bmp"` → `"monkeyskin"`.
pub fn texture_stem(raw: &str) -> String {
    let clean = raw
        .trim_start_matches(".\\")
        .trim_start_matches("./")
        .replace('\\', "/");
    Path::new(&clean)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
