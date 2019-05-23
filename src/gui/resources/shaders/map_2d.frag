//"#version 100\n"  // OpenGL ES 2.0
//"#version 120\n"  // OpenGL 2.1

varying highp vec3 f_color;

void main(void) {
    gl_FragColor = vec4(f_color.r, f_color.g, f_color.b, 1.0);
}
