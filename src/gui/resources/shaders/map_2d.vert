//"#version 100\n"  // OpenGL ES 2.0
//"#version 120\n"  // OpenGL 2.1

attribute vec3 coord3d;
attribute vec3 v_color;

varying vec3 f_color;
uniform mat4 MVP;

void main(void) {
  gl_Position = MVP * vec4(coord3d, 1.0);
  f_color = v_color;
}
