//"#version 100\n"  // OpenGL ES 2.0

uniform sampler2D texture;
varying highp vec2 tex_coords_0;

void main(void)
{
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture2D(texture, tex_coords_0).r);
    gl_FragColor = vec4(vec3(0.0, 0.0, 1.0), 1.0) * sampled;
}
