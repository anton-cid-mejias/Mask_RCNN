import tensorflow as tf
import numpy as np

# Copied from tensorflow graphics API
# https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py
def _build_matrix_from_sines_and_cosines(sin_angles, cos_angles):
  """Builds a rotation matrix from sines and cosines of Euler angles.
  Note:
    In the following, A1 to An are optional batch dimensions.
  Args:
    sin_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the sine of the Euler angles.
    cos_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the cosine of the Euler angles.
  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.
  """
  sin_angles.shape.assert_is_compatible_with(cos_angles.shape)

  sx, sy, sz = tf.unstack(sin_angles, axis=-1)
  cx, cy, cz = tf.unstack(cos_angles, axis=-1)
  m00 = cy * cz
  m01 = (sx * sy * cz) - (cx * sz)
  m02 = (cx * sy * cz) + (sx * sz)
  m10 = cy * sz
  m11 = (sx * sy * sz) + (cx * cz)
  m12 = (cx * sy * sz) - (sx * cz)
  m20 = -sy
  m21 = sx * cy
  m22 = cx * cy
  matrix = tf.stack((m00, m01, m02,
                     m10, m11, m12,
                     m20, m21, m22),
                    axis=-1)  # pyformat: disable
  output_shape = tf.concat((tf.shape(input=sin_angles)[:-1], (3, 3)), axis=-1)
  return tf.reshape(matrix, shape=output_shape)

def from_euler(angles, name=None):
  r"""Convert an Euler angle representation to a rotation matrix.
  The resulting matrix is $$\mathbf{R} = \mathbf{R}_z\mathbf{R}_y\mathbf{R}_x$$.
  Note:
    In the following, A1 to An are optional batch dimensions.
  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three Euler angles. `[A1, ..., An, 0]` is the angle about
      `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians and
      `[A1, ..., An, 2]` is the angle about `z` in radians.
    name: A name for this op that defaults to "rotation_matrix_3d_from_euler".
  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.
  Raises:
    ValueError: If the shape of `angles` is not supported.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_3d_from_euler", [angles]):
    angles = tf.convert_to_tensor(value=angles)

    sin_angles = tf.sin(angles)
    cos_angles = tf.cos(angles)
    return _build_matrix_from_sines_and_cosines(sin_angles, cos_angles)

def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    R = rotation_matrices
    sy = tf.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6
    singular = tf.cast(singular, tf.float32)

    x = tf.atan2(R[:, 2, 1], R[:, 2, 2])
    y = tf.atan2(-R[:, 2, 0], sy)
    z = tf.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = tf.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = tf.atan2(-R[:, 2, 0], sy)
    zs = R[:, 1, 0] * 0

    x = x * (1 - singular) + xs * singular
    y = y * (1 - singular) + ys * singular
    z = z * (1 - singular) + zs * singular

    out_euler = tf.stack([x, y, z], axis=1)

    return out_euler

# Input: batch*n
# Output: batch*n normalize
def normalize_vector(v, return_mag=False):
    v_mag = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(v, 2), axis=1))
    v_mag = tf.math.maximum(v_mag, tf.constant(1e-8))
    v_mag = tf.repeat(tf.reshape(v_mag, (-1, 1)), v.shape[1], axis=1)
    v = v / v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v

def cross_product(u, v):
    #batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = tf.concat((tf.reshape(i, (-1, 1)), tf.reshape(j, (-1, 1)), tf.reshape(k, (-1, 1))), 1)
    return out

def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = tf.reshape(x, (-1, 3, 1))
    y = tf.reshape(y, (-1, 3, 1))
    z = tf.reshape(z, (-1, 3, 1))
    matrix = tf.concat((x, y, z), 2)  # batch*3*3
    return matrix

def compute_geodesic_distance_from_two_matrices(m1, m2):
    # Remove batch dimension
    m1 = tf.reshape(m1, (-1, 3))
    m2 = tf.reshape(m2, (-1, 3))

    m = tf.matmul(m1, tf.transpose(m2))

    m = tf.reshape(m, (-1, 3, 3))

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = tf.clip_by_value(cos, clip_value_min=-1, clip_value_max=1)

    theta = tf.math.acos(cos)

    return theta

def test():
    # Test normalize vector
    v = tf.constant([[5., 0., 0.], [2., 2., 4.], [0., 0., 7.]])
    normalized = normalize_vector(v)
    sess = tf.compat.v1.Session()
    normalized = sess.run(normalized)
    print(normalized)
    
    # Test cross product
    v = tf.constant([[1, 2, 3]])
    u = tf.constant([[4, 5, 6]])
    cross = cross_product(u, v)
    cross = sess.run(cross)
    print(cross)

    # Rotation matrix from ortho 6d
    v = tf.constant([[1., 2., 3., 4., 5., 6.]])
    rm = compute_rotation_matrix_from_ortho6d(v)
    rm = sess.run(rm)
    print(rm)

    # Rotation matrix from euler ZYX
    angles = np.radians(np.array([180., 20., 70.]))
    euler = tf.constant(angles)
    rm = from_euler(euler)
    rm = sess.run(rm)
    print(rm)

    # Euler from rotation matrix
    euler = compute_euler_angles_from_rotation_matrices(rm.reshape(1, 3, 3))
    euler = sess.run(euler)
    print(np.degrees(euler))

    # Geodesic error
    angles1 = np.radians(np.array([33., -11., 79.]).astype(np.float))
    euler = tf.constant(angles1)
    rm1 = from_euler(euler)
    angles2 = np.radians(np.array([32., 7.51, -109]).astype(np.float))
    euler = tf.constant(angles2)
    rm2 = from_euler(euler)
    error = compute_geodesic_distance_from_two_matrices(rm1, rm2)
    error = sess.run(error)
    print(error)

if __name__=="__main__":
    test()