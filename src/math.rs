/// Row-major 4×4 matrix: `m[row][col]`.
/// Standard convention: translation in column 3 (`m[0..2][3]`),
/// homogeneous row in row 3 (`m[3][0..3] = [0,0,0,1]`).
pub type Mat4 = [[f32; 4]; 4];

/// Convert a B3D position `[x, y, z]` (forward = -Z) to glTF (forward = +Y)
/// by swapping the Y and Z axes.
pub fn neg_z_pos(p: [f32; 3]) -> [f32; 3] {
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
pub fn neg_z_quat(q: [f32; 4]) -> [f32; 4] {
    [q[0], q[1], q[3], q[2]]
}

/// Root-bone rotation: apply -90° X to the converted quaternion.
/// q = q(-90° X) * neg_z_quat(q_b3d)
pub fn root_quat(q: [f32; 4]) -> [f32; 4] {
    let q_conv = neg_z_quat(q);
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
/// (via `neg_z_pos` / `neg_z_quat`).
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
