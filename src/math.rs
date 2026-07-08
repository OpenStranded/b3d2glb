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

/// Row-major 4×4 matrix: `m[row][col]`.
/// Standard convention: translation in column 3 (`m[0..2][3]`),
/// homogeneous row in row 3 (`m[3][0..3] = [0,0,0,1]`).
pub type Mat4 = [[f32; 4]; 4];

/// Convert a B3D position `[x, y, z]` (left-handed Y-up, forward = -Z) to
/// glTF (right-handed Y-up, forward = +Y) by swapping the Y and Z axes.
pub fn swap_yz_pos(p: [f32; 3]) -> [f32; 3] {
    [p[0], p[2], p[1]]
}

/// Root-bone position: negate Z instead of swapping YZ.
/// B3D root bone position → glTF: [x, y, -z].
pub fn root_pos(p: [f32; 3]) -> [f32; 3] {
    [p[0], p[1], -p[2]]
}

/// Convert a B3D quaternion `[w, x, y, z]` (left-handed Y-up) to right-handed
/// Y-up by swapping the Y/Z components of the rotation axis.
/// Result is still `[w, x, y, z]`; call sites reorder to glTF's `[x, y, z, w]`.
pub fn swap_yz_quat(q: [f32; 4]) -> [f32; 4] {
    [q[0], q[1], q[3], q[2]]
}

/// Root-bone rotation: apply -90° X to the converted quaternion.
/// q = q(-90° X) * swap_yz_quat(q_b3d)
pub fn root_quat(q: [f32; 4]) -> [f32; 4] {
    let q_conv = swap_yz_quat(q);
    // q(-90° X) = [cos(-45°), sin(-45°), 0, 0] = [0.7071, -0.7071, 0, 0]
    let q_rot = [0.70710677, -0.70710677, 0.0, 0.0];
    quat_mul(&q_rot, &q_conv)
}

/// Convert internal `[w, x, z, y]` to glTF storage order `[x, z, y, w]`.
pub fn quat_to_gltf(q: [f32; 4]) -> [f32; 4] {
    [q[1], q[2], q[3], q[0]]
}
/// Both inputs and output are `[w, x, y, z]`.
pub fn quat_mul(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    let (w1, x1, y1, z1) = (a[0], a[1], a[2], a[3]);
    let (w2, x2, y2, z2) = (b[0], b[1], b[2], b[3]);
    [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ]
}

/// Build a TRS matrix from B3D bind-pose data.
///
/// `pos` and `rot` should already be converted to right-handed Y-up
/// (via `swap_yz_pos` / `swap_yz_quat`).
///
/// The rotation submatrix uses **column-major** formulas so that the matrix
/// matches what glTF's TRS reconstruction produces. This ensures
/// `node_matrix × IBM = I` at bind time.
///
/// Storage convention: `m[row][col]`, translation in `m[0..2][3]`.
pub fn b3d_to_mat4(pos: [f32; 3], scale: [f32; 3], rot: [f32; 4]) -> Mat4 {
    let (x, y, z, w) = (rot[1], rot[2], rot[3], rot[0]);
    let xx = x * x; let yy = y * y; let zz = z * z;
    let xy = x * y; let xz = x * z; let yz = y * z;
    let wx = w * x; let wy = w * y; let wz = w * z;

    let mut m = [[0.0f32; 4]; 4];

    // Diagonal: same in both row-major and column-major
    m[0][0] = (1.0 - 2.0 * (yy + zz)) * scale[0];
    m[1][1] = (1.0 - 2.0 * (xx + zz)) * scale[1];
    m[2][2] = (1.0 - 2.0 * (xx + yy)) * scale[2];

    // Off-diagonals: column-major convention (matches glTF TRS)
    m[0][1] = 2.0 * (xy - wz) * scale[1];
    m[0][2] = 2.0 * (xz + wy) * scale[2];
    m[1][0] = 2.0 * (xy + wz) * scale[0];
    m[1][2] = 2.0 * (yz - wx) * scale[2];
    m[2][0] = 2.0 * (xz - wy) * scale[0];
    m[2][1] = 2.0 * (yz + wx) * scale[1];

    // Translation row
    m[0][3] = pos[0];
    m[1][3] = pos[1];
    m[2][3] = pos[2];

    // Homogeneous row
    m[3][0] = 0.0;
    m[3][1] = 0.0;
    m[3][2] = 0.0;
    m[3][3] = 1.0;

    m
}

/// Row-major 4×4 matrix multiply: `r = a * b`.
pub fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut r = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            r[i][j] = a[i][0] * b[0][j]
                     + a[i][1] * b[1][j]
                     + a[i][2] * b[2][j]
                     + a[i][3] * b[3][j];
        }
    }
    r
}

/// Inverse of a row-major 4×4 matrix using cofactors.
pub fn mat4_inverse(m: &Mat4) -> Mat4 {
    let (m00, m01, m02, m03) = (m[0][0], m[0][1], m[0][2], m[0][3]);
    let (m10, m11, m12, m13) = (m[1][0], m[1][1], m[1][2], m[1][3]);
    let (m20, m21, m22, m23) = (m[2][0], m[2][1], m[2][2], m[2][3]);
    let (m30, m31, m32, m33) = (m[3][0], m[3][1], m[3][2], m[3][3]);

    let a = m00*m11 - m01*m10; let b = m00*m12 - m02*m10;
    let c = m00*m13 - m03*m10; let d = m01*m12 - m02*m11;
    let e = m01*m13 - m03*m11; let f = m02*m13 - m03*m12;
    let g = m20*m31 - m21*m30; let h = m20*m32 - m22*m30;
    let i = m20*m33 - m23*m30; let j = m21*m32 - m22*m31;
    let k = m21*m33 - m23*m31; let l = m22*m33 - m23*m32;

    let det = a*l - b*k + c*j + d*i - e*h + f*g;
    if det == 0.0 { return [[0.0; 4]; 4]; }
    let inv_det = 1.0 / det;

    let mut inv = [[0.0; 4]; 4];
    inv[0][0] = ( m11*l - m12*k + m13*j) * inv_det;
    inv[0][1] = (-m01*l + m02*k - m03*j) * inv_det;
    inv[0][2] = ( m31*f - m32*e + m33*d) * inv_det;
    inv[0][3] = (-m21*f + m22*e - m23*d) * inv_det;
    inv[1][0] = (-m10*l + m12*i - m13*h) * inv_det;
    inv[1][1] = ( m00*l - m02*i + m03*h) * inv_det;
    inv[1][2] = (-m30*f + m32*c - m33*b) * inv_det;
    inv[1][3] = ( m20*f - m22*c + m23*b) * inv_det;
    inv[2][0] = ( m10*k - m11*i + m13*g) * inv_det;
    inv[2][1] = (-m00*k + m01*i - m03*g) * inv_det;
    inv[2][2] = ( m30*e - m31*c + m33*a) * inv_det;
    inv[2][3] = (-m20*e + m21*c - m23*a) * inv_det;
    inv[3][0] = (-m10*j + m11*h - m12*g) * inv_det;
    inv[3][1] = ( m00*j - m01*h + m02*g) * inv_det;
    inv[3][2] = (-m30*d + m31*b - m32*a) * inv_det;
    inv[3][3] = ( m20*d - m21*b + m22*a) * inv_det;
    inv
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 2e-5;
    const C: f32 = 0.70710677; // cos(45°) / sin(45°)

    fn assert_mat4_eq(a: &Mat4, b: &Mat4) {
        for i in 0..4 {
            for j in 0..4 {
                let diff = (a[i][j] - b[i][j]).abs();
                assert!(diff < EPS, "mismatch at [{i}][{j}]: {} vs {}", a[i][j], b[i][j]);
            }
        }
    }

    fn identity() -> Mat4 {
        [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]]
    }

    #[test]
    fn test_identity_matrix() {
        let m = b3d_to_mat4([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]);
        assert_mat4_eq(&m, &identity());
    }

    #[test]
    fn test_translation_only() {
        let m = b3d_to_mat4([10.0, -5.0, 3.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]);
        let expected = [
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, -5.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_mat4_eq(&m, &expected);
    }

    #[test]
    fn test_scale_only() {
        // Scale (2,3,4) should produce diagonal [2,3,4,1]
        let m = b3d_to_mat4([0.0, 0.0, 0.0], [2.0, 3.0, 4.0], [1.0, 0.0, 0.0, 0.0]);
        let expected = [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_mat4_eq(&m, &expected);
    }

    #[test]
    fn test_rotation_180_x() {
        // 180° around X: quat (w=0, x=1, y=0, z=0)
        // Column-major rotation: x unchanged, y/z flipped
        let m = b3d_to_mat4([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]);
        let expected = [
            [1.0,  0.0,  0.0, 0.0],
            [0.0, -1.0,  0.0, 0.0],
            [0.0,  0.0, -1.0, 0.0],
            [0.0,  0.0,  0.0, 1.0],
        ];
        assert_mat4_eq(&m, &expected);
    }

    #[test]
    fn test_rotation_90_y() {
        // 90° around Y: quat (w=0.7071, x=0, y=0.7071, z=0)
        // Rotates +X to +Z (right-handed)
        let m = b3d_to_mat4([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [C, 0.0, C, 0.0]);
        // Column-major: [cos=0, 0, sin=1; 0,1,0; -sin=-1,0,cos=0]
        let expected = [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        assert_mat4_eq(&m, &expected);
    }

    #[test]
    fn test_mat4_inverse_identity() {
        let inv = mat4_inverse(&identity());
        assert_mat4_eq(&inv, &identity());
    }

    #[test]
    fn test_mat4_inverse_translation() {
        let m = b3d_to_mat4([5.0, -3.0, 2.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]);
        let inv = mat4_inverse(&m);
        // Inverse should negate translation
        let expected = b3d_to_mat4([-5.0, 3.0, -2.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]);
        assert_mat4_eq(&inv, &expected);
    }

    #[test]
    fn test_mat4_inverse_rotation() {
        let q = [0.5, 0.5, 0.5, 0.5]; // arbitrary unit quat
        let m = b3d_to_mat4([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], q);
        let inv = mat4_inverse(&m);
        // Inverse should be transpose (since rotation matrices are orthogonal)
        let mut expected = [[0.0; 4]; 4];
        for i in 0..3 {
            for j in 0..3 {
                expected[i][j] = m[j][i];
            }
        }
        expected[3][3] = 1.0;
        assert_mat4_eq(&inv, &expected);
    }

    #[test]
    fn test_mat4_mul_identity() {
        let a = identity();
        let b = b3d_to_mat4([1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [1.0, 0.0, 0.0, 0.0]);
        assert_mat4_eq(&mat4_mul(&a, &b), &b);
        assert_mat4_eq(&mat4_mul(&b, &a), &b);
    }

    #[test]
    fn test_matrix_times_inverse_is_identity() {
        let test_cases = [
            // (pos, scale, quat)
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]),
            ([5.0, -3.0, 2.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [C, 0.0, C, 0.0]),
            ([10.0, 20.0, -5.0], [1.0, 1.0, 1.0], [0.7071, 0.0, 0.0, 0.7071]),
            ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.9239, 0.3827, 0.0, 0.0]), // 45° X
        ];
        for (pos, scale, rot) in &test_cases {
            let m = b3d_to_mat4(*pos, *scale, *rot);
            let inv = mat4_inverse(&m);
            let product = mat4_mul(&m, &inv);
            assert_mat4_eq(&product, &identity());
        }
    }

    /// CRITICAL: The IBM must satisfy joint_matrix = node_matrix * IBM = I
    /// at bind time. This test verifies that b3d_to_mat4 builds a matrix
    /// whose inverse, when multiplied on the right, yields identity.
    #[test]
    fn test_joint_matrix_is_identity_at_bind() {
        // Simulate a root bone with position + rotation.
        let b3d_pos = [0.0333, 17.5667, -6.0212];
        let b3d_rot = [1.0, 0.0, 0.0, 0.0];
        let scale = [1.0, 1.0, 1.0];

        let (gltf_pos, gltf_rot) = (root_pos(b3d_pos), root_quat(b3d_rot));
        // glTF node matrix = b3d_to_mat4(gltf_pos, scale, gltf_rot)
        let node_mat = b3d_to_mat4(gltf_pos, scale, gltf_rot);
        // IBM = inv(b3d_to_mat4(gltf_pos, scale, gltf_rot))
        let ibm = mat4_inverse(&node_mat);
        // joint_matrix = node_mat * ibm should be I
        let joint_mat = mat4_mul(&node_mat, &ibm);
        assert_mat4_eq(&joint_mat, &identity());
    }

    #[test]
    fn test_swap_yz_pos() {
        assert_eq!(swap_yz_pos([1.0, 2.0, 3.0]), [1.0, 3.0, 2.0]);
        assert_eq!(swap_yz_pos([0.0, -5.0, 10.0]), [0.0, 10.0, -5.0]);
    }

    #[test]
    fn test_root_pos() {
        assert_eq!(root_pos([1.0, 2.0, 3.0]), [1.0, 2.0, -3.0]);
        assert_eq!(root_pos([0.0, 5.0, -10.0]), [0.0, 5.0, 10.0]);
    }

    #[test]
    fn test_swap_yz_quat() {
        // [w, x, y, z] → [w, x, z, y] (swap y/z components)
        assert_eq!(swap_yz_quat([1.0, 0.0, 0.0, 0.0]), [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(swap_yz_quat([0.0, 1.0, 2.0, 3.0]), [0.0, 1.0, 3.0, 2.0]);
    }

    #[test]
    fn test_quat_to_gltf() {
        // Internal [w, x, z, y] → glTF [x, z, y, w]
        assert_eq!(quat_to_gltf([1.0, 0.0, 0.0, 0.0]), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(quat_to_gltf([0.0, 1.0, 0.0, 0.0]), [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_quat_mul_identity() {
        let q = [0.7071, 0.7071, 0.0, 0.0];
        assert_eq!(quat_mul(&[1.0, 0.0, 0.0, 0.0], &q), q);
        assert_eq!(quat_mul(&q, &[1.0, 0.0, 0.0, 0.0]), q);
    }

    #[test]
    fn test_root_quat_applies_minus_90_x() {
        // root_quat = q(-90°X) × swap_yz_quat(identity) = q(-90°X) × identity
        // q(-90°X) = [cos(-45°), sin(-45°), 0, 0] = [0.7071, -0.7071, 0, 0]
        let r = root_quat([1.0, 0.0, 0.0, 0.0]);
        // quat_mul(&a, &b) = a * b
        // root_quat = q_rot * swap_yz_quat(q_b3d)
        // q_rot = [0.70710677, -0.70710677, 0.0, 0.0]
        // swap_yz_quat(identity) = [1.0, 0.0, 0.0, 0.0]
        // result = quat_mul([0.7071, -0.7071, 0, 0], [1, 0, 0, 0]) = [0.7071, -0.7071, 0, 0]
        assert!((r[0] - 0.70710677).abs() < EPS);
        assert!((r[1] + 0.70710677).abs() < EPS);
        assert!((r[2]).abs() < EPS);
        assert!((r[3]).abs() < EPS);
    }

    #[test]
    fn test_neg_z_rotation_off_axis() {
        // For a non-identity rotation, swap_yz_quat swaps y/z components
        let q = [0.7071, 0.0, 0.7071, 0.0]; // 90° around Y in [w,x,y,z]
        let qn = swap_yz_quat(q); // [w, x, z, y] = [0.7071, 0.0, 0.0, 0.7071]
        assert!((qn[0] - 0.7071).abs() < EPS);
        assert!((qn[1]).abs() < EPS);
        assert!((qn[2]).abs() < EPS);
        assert!((qn[3] - 0.7071).abs() < EPS);
    }
}
